"""Molecule-only batch interfaces for GA energy evaluation and optimization."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

from molecular_qm_models import Molecule, MoleculeList
from pydantic import BaseModel, Field, PrivateAttr

from molecular_qm_search.optimization.models.ga_models import GAOptimizationMethod


class MoleculeEvaluator(Protocol):
    """Return one new molecule per input, in input order and atom order.

    Results must carry finite ``properties['energy']`` values in kcal/mol.
    ``score`` preserves geometry; ``optimize`` returns the evaluated geometry.
    Implementations must not mutate the input molecules.
    """

    def score(self, molecules: list[Molecule]) -> list[Molecule]: ...

    def optimize(self, molecules: list[Molecule], *, max_iters: int) -> list[Molecule]: ...


def validate_results(inputs: list[Molecule], results: list[Molecule]) -> list[Molecule]:
    results = list(results)
    if len(inputs) != len(results):
        raise ValueError("Evaluator must return one molecule per input, in input order")
    for index, (source, result) in enumerate(zip(inputs, results)):
        if not isinstance(result, Molecule):
            raise TypeError(f"Evaluator result {index} is not a Molecule")
        if [a.element for a in source.atoms] != [a.element for a in result.atoms]:
            raise ValueError(f"Evaluator changed the atom sequence for molecule {index}")
        energy = result.properties.get("energy")
        if energy is None or not math.isfinite(float(energy)):
            raise ValueError(f"Evaluator result {index} has no finite energy")
        if result.properties.get("energy_unit", "kcal/mol") != "kcal/mol":
            raise ValueError("Evaluator energies must be converted to kcal/mol")
    return results


def _agent_debug(hypothesis: str, message: str, data: dict) -> None:
    # #region agent log
    row = {
        "sessionId": "2d4a99",
        "hypothesisId": hypothesis,
        "location": "ga_evaluation.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with Path(r"C:\Users\bj7610\WebstormProjects\simstack_ui-ww-new-features\debug-2d4a99.log").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        return
    # #endregion


class RDKitEvaluator(BaseModel):
    forcefield: str = "mmff"
    threads: int = Field(default=0, ge=0)
    _connectivity: Optional[Molecule] = PrivateAttr(default=None)
    _bond_template: object = PrivateAttr(default=None)

    def bind_connectivity(self, molecule: Molecule) -> None:
        """Score later conformers on this molecule's bonds, not a distance guess."""
        if molecule is None or not molecule.atoms:
            raise ValueError("RDKit evaluation requires the input molecule's connectivity")
        self._connectivity = molecule
        self._bond_template = None

    def score(self, molecules: list[Molecule]) -> list[Molecule]:
        if self._connectivity is None:
            from molecular_qm_util import evaluate_molecules_rdkit

            return evaluate_molecules_rdkit(molecules, self.forcefield, threads=self.threads)
        return self._evaluate_on_connectivity(
            molecules, optimize=False, max_iters=500, attempts=1, require_improvement=False,
        )

    def optimize(
            self,
            molecules: list[Molecule],
            *,
            max_iters: int,
            attempts: int = 3,
            require_improvement: bool = False,
    ) -> list[Molecule]:
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError(f"attempts must be a positive integer, got {attempts!r}")
        if not isinstance(require_improvement, bool):
            raise ValueError(
                f"require_improvement must be a bool, got {require_improvement!r}"
            )
        if self._connectivity is None:
            from molecular_qm_util import evaluate_molecules_rdkit

            return evaluate_molecules_rdkit(
                molecules,
                self.forcefield,
                optimize=True,
                max_iters=max_iters,
                threads=self.threads,
            )
        return self._evaluate_on_connectivity(
            molecules,
            optimize=True,
            max_iters=max_iters,
            attempts=attempts,
            require_improvement=require_improvement,
        )

    def _template(self):
        if self._bond_template is None:
            from molecular_qm_util.rdkit_scripts.rdkit_optimize import molecule_to_rdkit

            self._bond_template = molecule_to_rdkit(self._connectivity)
        return self._bond_template

    def _evaluate_on_connectivity(
            self, molecules, *, optimize: bool, max_iters: int, attempts: int, require_improvement: bool,
    ):
        import os
        from concurrent.futures import ThreadPoolExecutor

        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D

        template = self._template()
        name = self.forcefield.lower().removeprefix("rdkit_")
        name = "mmff94" if name == "mmff" else name
        if name not in {"uff", "mmff94", "mmff94s"}:
            raise ValueError(f"Unsupported force field: {self.forcefield!r}")

        def bonds_of(rdkit_mol):
            found = set()
            for bond in rdkit_mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                found.add((min(begin, end), max(begin, end)))
            return found

        seed_bonds = bonds_of(template)

        def evaluate(molecule: Molecule) -> Molecule:
            if len(molecule.atoms) != template.GetNumAtoms():
                raise ValueError(
                    f"Conformer has {len(molecule.atoms)} atoms; the input molecule has {template.GetNumAtoms()}"
                )
            rdkit_mol = Chem.Mol(template)
            conformer = rdkit_mol.GetConformer()
            for index, atom in enumerate(molecule.atoms):
                if rdkit_mol.GetAtomWithIdx(index).GetSymbol() != atom.element:
                    raise ValueError(
                        f"Atom {index + 1} is {atom.element} in the conformer and "
                        f"{rdkit_mol.GetAtomWithIdx(index).GetSymbol()} in the input molecule"
                    )
                conformer.SetAtomPosition(index, Point3D(float(atom.x), float(atom.y), float(atom.z)))
            if name == "uff":
                if not AllChem.UFFHasAllMoleculeParams(rdkit_mol):
                    raise ValueError("UFF parameters are unavailable for this molecule")
                force_field = AllChem.UFFGetMoleculeForceField(rdkit_mol)
            else:
                variant = "MMFF94s" if name == "mmff94s" else "MMFF94"
                properties = AllChem.MMFFGetMoleculeProperties(rdkit_mol, mmffVariant=variant)
                if properties is None:
                    raise ValueError(f"{variant} parameters are unavailable for this molecule")
                force_field = AllChem.MMFFGetMoleculeForceField(rdkit_mol, properties)
            if force_field is None:
                raise ValueError(f"Cannot construct {name} force field")
            force_field.Initialize()
            # Same force tolerance Minimize uses. RDKit also returns success when the
            # step collapses, which leaves the raw gradient large.
            force_tol = 1e-4
            discarded = False
            gradient_test = None
            accepted_energy = None
            progress_rows = []
            if optimize:
                performed = 0
                minimize_force_tol = force_tol
                rdkit_converged = False
                met = False
                for _attempt in range(attempts):
                    if require_improvement and rdkit_converged:
                        minimize_force_tol = minimize_force_tol / 10.0
                        if not math.isfinite(minimize_force_tol) or minimize_force_tol <= 0:
                            raise ValueError(
                                "Force tolerance became "
                                f"{minimize_force_tol} while repeating the initial relaxation"
                            )
                        force_field.Initialize()
                        rdkit_converged = False
                    status = force_field.Minimize(
                        maxIts=int(max_iters), forceTol=minimize_force_tol, energyTol=1e-6,
                    )
                    performed += 1
                    raw_gradient = tuple(force_field.CalcGrad())
                    if any(not math.isfinite(component) for component in raw_gradient):
                        raise ValueError("Force-field gradient is not finite")
                    if len(raw_gradient) != conformer.GetNumAtoms() * 3:
                        raise ValueError(
                            f"Force field returned {len(raw_gradient)} gradient components "
                            f"for {conformer.GetNumAtoms()} atoms"
                        )
                    gradient_norm = math.sqrt(sum(component * component for component in raw_gradient))
                    if not math.isfinite(gradient_norm):
                        raise ValueError("Force-field gradient norm is not finite")
                    scaled = [component * 0.1 for component in raw_gradient]
                    max_component = max(abs(component) for component in scaled)
                    grad_scale = 0.1
                    if max_component > 10.0:
                        while max_component * grad_scale > 10.0:
                            grad_scale *= 0.5
                        scaled = [component * grad_scale for component in scaled]
                    energy = float(force_field.CalcEnergy())
                    if not math.isfinite(energy):
                        raise ValueError("Force-field energy is not finite")
                    term = max(energy * grad_scale, 1.0)
                    gradient_test = 0.0
                    for component_index, component in enumerate(scaled):
                        position = conformer.GetAtomPosition(component_index // 3)
                        coordinate = (position.x, position.y, position.z)[component_index % 3]
                        gradient_test = max(
                            gradient_test, abs(component) * max(abs(coordinate), 1.0)
                        )
                    gradient_test /= term
                    if not math.isfinite(gradient_test):
                        raise ValueError("Force-field gradient is not finite")
                    accepted_energy = energy
                    if require_improvement:
                        progress_rows.append(
                            f"Initial relaxation {performed}/{attempts}: "
                            f"energy {energy:.6g} kcal/mol, gradient norm {gradient_norm:.6g}"
                        )
                    if gradient_test < force_tol:
                        met = True
                        break
                    if (
                        require_improvement
                        and isinstance(status, int)
                        and not isinstance(status, bool)
                        and status == 0
                    ):
                        rdkit_converged = True
                discarded = not met
                if performed < 1 or accepted_energy is None:
                    raise ValueError("Force-field optimization finished without an accepted step")
            result = Molecule.from_molecule(molecule)
            result.properties = copy.deepcopy(molecule.properties)
            result.properties.pop("rank_id", None)
            result.properties.pop("optimization_converged", None)
            result.properties.pop("force_field_discarded", None)
            result.properties.pop("force_field_gradient", None)
            result.properties.pop("force_field_attempts", None)
            result.properties.pop("force_field_progress", None)
            if optimize:
                for index, atom in enumerate(result.atoms):
                    position = conformer.GetAtomPosition(index)
                    atom.position = [position.x, position.y, position.z]
                result.properties["optimization_converged"] = not discarded
                if discarded:
                    if gradient_test is None:
                        raise ValueError("Force-field optimization stopped without a gradient")
                    result.properties["force_field_discarded"] = True
                    result.properties["force_field_gradient"] = float(gradient_test)
                if require_improvement:
                    result.properties["force_field_attempts"] = performed
                    if not progress_rows:
                        raise ValueError("Initial relaxation finished without a progress record")
                    result.properties["force_field_progress"] = "\n".join(progress_rows)
            reported_energy = accepted_energy if optimize else float(force_field.CalcEnergy())
            result.properties.update(
                energy=float(reported_energy),
                energy_unit="kcal/mol",
                method=f"rdkit+{name}",
            )
            return result

        if not molecules:
            return []
        workers = self.threads or max(1, (os.cpu_count() or 1) // 2)
        with ThreadPoolExecutor(max_workers=min(workers, len(molecules))) as executor:
            results = list(executor.map(evaluate, molecules))

        # #region agent log
        from molecular_qm_util.rdkit_scripts.rdkit_optimize import molecule_to_rdkit

        mismatches = 0
        for molecule in molecules:
            perceived = bonds_of(molecule_to_rdkit(molecule))
            if perceived != seed_bonds:
                mismatches += 1
        _agent_debug("A", "RDKit evaluation on the input bond graph", {
            "optimize": optimize,
            "conformers": len(results),
            "bond_graph_mismatches": mismatches,
            "seed_bonds": len(seed_bonds),
            "min_energy": min(float(item.properties["energy"]) for item in results),
            "max_energy": max(float(item.properties["energy"]) for item in results),
        })
        # #endregion
        return results


class CallableEvaluator(BaseModel):
    """Adapt external batch APIs to the GA.

    Callbacks implement the MoleculeEvaluator contract, including kcal/mol units.
    The optimization callback accepts ``max_iters`` as a keyword argument.
    """

    score_batch: Callable[[list[Molecule]], list[Molecule]]
    optimize_batch: Callable[..., list[Molecule]]

    def score(self, molecules: list[Molecule]) -> list[Molecule]:
        return self.score_batch(molecules)

    def optimize(self, molecules: list[Molecule], *, max_iters: int) -> list[Molecule]:
        return self.optimize_batch(molecules, max_iters=max_iters)


class NodeMoleculeEvaluator:
    """Bridge synchronous batch calls to asynchronous SimStack child nodes.

    The GA's synchronous loop runs in a worker thread when called from an async
    node; child calculations are submitted back to that node's event loop.
    Calls are serial so inline calculators never compete for their output files.
    """

    HARTREE_TO_KCAL_MOL = 627.5094740631

    def __init__(self, options=None, *, loop=None, node_kwargs=None):
        self.options = dict(options or {})
        self.loop = loop
        self.node_kwargs = dict(node_kwargs or {})
        self.node_kwargs.pop("node_runner", None)

    def _run(self, molecules, *, optimize, max_iters=0):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None:
            raise RuntimeError("Run synchronous GA/NodeMoleculeEvaluator calls with asyncio.to_thread")
        if self.loop is not None and not self.loop.is_running():
            raise RuntimeError("The calculator submission event loop is not running")
        job = self._evaluate(molecules, optimize=optimize, max_iters=max_iters)
        if self.loop is None:
            return asyncio.run(job)
        return asyncio.run_coroutine_threadsafe(job, self.loop).result()

    def score(self, molecules: list[Molecule]) -> list[Molecule]:
        return self._run(molecules, optimize=False)

    def optimize(self, molecules: list[Molecule], *, max_iters: int) -> list[Molecule]:
        return self._run(molecules, optimize=True, max_iters=max_iters)


class DFTBEvaluator(NodeMoleculeEvaluator):
    """Evaluate a batch with molecular_qm_dftb's dftb_list_calculator node.

    One list-calculator call covers the whole batch. A per-molecule index is
    stored only on the submitted copies so identical geometries stay distinct
    inside MassRunner; it is removed from returned properties.
    """

    _BATCH_INDEX = "ga_dftb_batch_index"

    async def _evaluate(self, molecules, *, optimize, max_iters):
        from molecular_qm_dftb.models.dftb_input import DftbInput
        from molecular_qm_dftb.nodes.dftb_list_calculator import dftb_list_calculator

        options = dict(self.options)
        options.update(optimization=optimize, max_optimization_steps=max_iters,
                       tolerate_failure=False)
        opts = DftbInput(**options)
        batch = MoleculeList()
        submitted = []
        for index, molecule in enumerate(molecules):
            copied = Molecule.from_molecule(molecule)
            copied.properties = copy.deepcopy(molecule.properties)
            copied.properties[self._BATCH_INDEX] = index
            batch.add_molecule(copied)
            submitted.append(copied)
        output = await dftb_list_calculator(batch, opts, **self.node_kwargs)
        dataset = None if output is None else getattr(output, "dataset", None)
        if dataset is None and output is not None and (
            isinstance(output, dict) or hasattr(output, "sections")
        ) and "results" in output:
            dataset = output
        if dataset is None:
            raise RuntimeError("dftb_list_calculator did not return a dataset")
        if "results" not in dataset:
            raise RuntimeError("dftb_list_calculator dataset has no results section")
        rows = dataset["results"]
        if (
            hasattr(rows, "data")
            and rows.data
            and hasattr(rows, "load_to_cache")
            and len(rows) == 0
        ):
            from simstack.core.context import context

            await rows.load_to_cache(context.db)

        rows_by_id = {}
        for _row_name, row in rows.items():
            source = row.get("arg_molecule")
            if source is None or getattr(source, "id", None) is None:
                raise RuntimeError("DFTB list result row is missing the input molecule")
            if source.id in rows_by_id:
                raise RuntimeError("DFTB list returned duplicate rows for one molecule")
            rows_by_id[source.id] = row
        if len(rows_by_id) != len(submitted):
            raise RuntimeError(
                f"DFTB list returned {len(rows_by_id)} rows for {len(submitted)} molecules"
            )

        results = []
        for molecule, submitted_molecule in zip(molecules, submitted):
            row = rows_by_id.get(submitted_molecule.id)
            if row is None:
                raise RuntimeError("DFTB list result is missing a submitted molecule")
            success = row.get("success")
            if success is None:
                raise RuntimeError("DFTB result row is missing success")
            success_value = success.value if hasattr(success, "value") else success
            if not success_value:
                error = row.get("error")
                if error is None:
                    raise RuntimeError("DFTB calculation failed")
                error_msg = error.value if hasattr(error, "value") else error
                raise RuntimeError(f"DFTB calculation failed: {error_msg}")
            qm_result = row.get("result_qm_result")
            if qm_result is None or qm_result.final_energy is None:
                raise RuntimeError("DFTB returned no energy")
            if qm_result.normal_termination is False or qm_result.scf_converged is False:
                raise RuntimeError("DFTB calculation did not complete successfully")
            structure = qm_result.final_structure if optimize else molecule
            if structure is None:
                raise RuntimeError("DFTB optimization returned no final structure")
            result = Molecule.from_molecule(structure)
            result.properties = copy.deepcopy(molecule.properties)
            structure_properties = copy.deepcopy(structure.properties)
            structure_properties.pop(self._BATCH_INDEX, None)
            result.properties.update(structure_properties)
            result.properties.pop("rank_id", None)
            result.properties.pop("optimization_converged", None)
            result.properties.update(
                energy=float(qm_result.final_energy) * self.HARTREE_TO_KCAL_MOL,
                energy_hartree=float(qm_result.final_energy),
                energy_unit="kcal/mol", method="dftb",
            )
            if optimize:
                result.properties["optimization_converged"] = qm_result.optimization_converged
            results.append(result)
        return validate_results(molecules, results)


class XTBEvaluator(NodeMoleculeEvaluator):
    """Evaluate batches using the xTB nodes in molecular_qm_psi4."""

    async def _evaluate(self, molecules, *, optimize, max_iters):
        from molecular_qm_psi4.models.crest_input import XTBInput
        from molecular_qm_psi4.nodes.crest import xtb_molecule_list, xtb_optimize_molecule_list

        batch = MoleculeList()
        for molecule in molecules:
            copied = Molecule.from_molecule(molecule)
            copied.properties = copy.deepcopy(molecule.properties)
            for key in ("energy", "energy_hartree", "energy_unit", "rank_id",
                        "optimization_converged"):
                copied.properties.pop(key, None)
            batch.add_molecule(copied)
        options = dict(self.options)
        options.update(molecules=batch, optimize=optimize)
        if optimize:
            options["max_iters"] = max_iters
        opts = XTBInput(**options)
        calculator = xtb_optimize_molecule_list if optimize else xtb_molecule_list
        output = await calculator(opts, **self.node_kwargs)
        returned = getattr(output, "molecule_list", None)
        if returned is None or getattr(output, "error_message", None):
            raise RuntimeError(f"xTB returned no complete batch: {getattr(output, 'error_message', '')}")
        results = []
        for molecule in returned:
            energy = molecule.properties.get("energy")
            if energy is None:
                raise RuntimeError("xTB returned a molecule without an energy")
            result = Molecule.from_molecule(molecule)
            result.properties = copy.deepcopy(molecule.properties)
            result.properties.update(
                energy=float(energy) * self.HARTREE_TO_KCAL_MOL,
                energy_hartree=float(energy), energy_unit="kcal/mol", method="xtb",
            )
            results.append(result)
        return validate_results(molecules, results)


def make_evaluator(method, *, parallel_children=0,
                   backend_options=None, loop=None, node_kwargs=None) -> MoleculeEvaluator:
    if parallel_children < 0:
        raise ValueError("parallel_children cannot be negative")
    if isinstance(method, GAOptimizationMethod):
        method = method.value
    if method == GAOptimizationMethod.RDKIT_MMFF:
        forcefield = "mmff94"
    elif method == GAOptimizationMethod.RDKIT_MMFF94S:
        forcefield = "mmff94s"
    elif method == GAOptimizationMethod.RDKIT_UFF:
        forcefield = "uff"
    elif method == GAOptimizationMethod.DFTB:
        return DFTBEvaluator(backend_options, loop=loop, node_kwargs=node_kwargs)
    elif method == GAOptimizationMethod.XTB:
        return XTBEvaluator(backend_options, loop=loop, node_kwargs=node_kwargs)
    else:
        raise ValueError(f"Unsupported GA evaluation method: {method!r}")
    return RDKitEvaluator(forcefield=forcefield, threads=parallel_children)
