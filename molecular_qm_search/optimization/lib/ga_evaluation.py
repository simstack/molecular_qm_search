"""Molecule-only batch interfaces for GA energy evaluation and optimization."""
from __future__ import annotations

import asyncio
import copy
import math
from typing import Callable, Protocol

from molecular_qm_models import Molecule, MoleculeList
from pydantic import BaseModel, Field

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


class RDKitEvaluator(BaseModel):
    forcefield: str = "mmff"
    threads: int = Field(default=0, ge=0)

    def score(self, molecules: list[Molecule]) -> list[Molecule]:
        from molecular_qm_util import score_molecules_rdkit

        return score_molecules_rdkit(molecules, self.forcefield, threads=self.threads)

    def optimize(self, molecules: list[Molecule], *, max_iters: int) -> list[Molecule]:
        from molecular_qm_util import optimize_molecules_rdkit

        return optimize_molecules_rdkit(
            molecules, self.forcefield, max_iters=max_iters, threads=self.threads
        )


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
