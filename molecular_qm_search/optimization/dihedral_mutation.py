"""GA dihedral mutations and a DFTB relaxation of the resulting geometries."""
import copy

import numpy as np
from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_dftb.nodes.dftb_list_calculator import dftb_list_calculator
from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters

from molecular_qm_search.optimization.ga_population import PopulationGenerator
from molecular_qm_search.optimization.lib.ga_evaluation import DFTBEvaluator


def _wrapped_degrees(value, reference):
    return (value - reference + 180.0) % 360.0 - 180.0


def _positions(molecule):
    return np.array([[atom.x, atom.y, atom.z] for atom in molecule.atoms], dtype=float)


def mutated_dihedral_molecules(molecule: Molecule) -> list[Molecule]:
    """Apply GA torsion steps and return two mutants whose geometry matches those steps.

    Every discovered rotatable dihedral is shifted by the GA step of +30 degrees.
    The Cartesian structure must show that signed change, leave the rest of the
    molecule fixed, and keep distances to the rotation axis. ``reproduce`` is
    checked the same way for its random single-coordinate mutations.
    """
    from molecular_qm_util import get_rotatable_bonds

    if not molecule.atoms:
        raise ValueError("Cannot search rotatable dihedrals of a molecule with no atoms")
    coordinates = get_rotatable_bonds(
        molecule,
        match_double_bonds=True,
        min_value=-180.0,
        max_value=180.0,
    )
    if not coordinates.elements:
        raise ValueError("No rotatable dihedrals were found")
    for coord in coordinates.elements:
        if coord.bond_type is None or not hasattr(coord.bond_type, "value"):
            raise ValueError(f"Rotatable dihedral has no bond type: {coord.bond_type!r}")
        if len(coord.atom_indices) != 4:
            raise ValueError(f"Dihedral atom indices must have length 4, got {coord.atom_indices}")
        if not coord.moving_atoms:
            raise ValueError(f"Dihedral {tuple(coord.atom_indices)} has no moving atoms")
        if not coord.real_values:
            raise ValueError(f"Dihedral {tuple(coord.atom_indices)} has no computed angle")

    original = _positions(molecule)
    generator = PopulationGenerator(
        molecule,
        coordinates=coordinates,
        seed=1,
        mutation_rate=1.0,
        dihedral_interval=30.0,
        db_treatment="180+step",
    )
    template = copy.deepcopy(generator.coordinates)
    selected = []
    for index, source in enumerate(template.elements):
        before = source.get_actual_value(source.value)
        for delta in (30.0, -30.0):
            child = copy.deepcopy(template)
            generator._shift(child.elements[index], delta)
            target = child.elements[index].get_actual_value(child.elements[index].value)
            if abs(_wrapped_degrees(target, before) - delta) > 1e-6:
                raise ValueError(
                    f"Dihedral {index} stored angle changed by "
                    f"{_wrapped_degrees(target, before)} degrees, expected {delta}"
                )
            isolated = generator._copy_molecule(molecule)
            coord = copy.deepcopy(child.elements[index])
            coord.set(isolated, coord.value)
            measured = coord.real_values[0]
            if abs(_wrapped_degrees(measured, target)) > 1e-4:
                raise ValueError(
                    f"Dihedral {tuple(source.atom_indices)} measured {measured} degrees "
                    f"after a {delta} degree mutation, expected {target}"
                )
            moved = _positions(isolated)
            moving = set(coord.moving_atoms)
            for atom_index in range(len(molecule.atoms)):
                shift = float(np.linalg.norm(moved[atom_index] - original[atom_index]))
                if atom_index not in moving and shift > 1e-6:
                    raise ValueError(
                        f"Atom {atom_index} is outside the rotated fragment but moved by {shift}"
                    )
            a2, a3, a4 = coord.atom_indices[1], coord.atom_indices[2], coord.atom_indices[3]
            axis = original[a3] - original[a2]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 0:
                raise ValueError(f"Dihedral {index} has a zero-length rotation axis")
            axis_unit = axis / axis_norm
            if abs(float(np.linalg.norm(moved[a2] - moved[a3])) - axis_norm) > 1e-5:
                raise ValueError(f"Mutation changed the length of rotatable bond {a2}-{a3}")
            fragment_moved = False
            for atom_index in moving:
                if atom_index == a3:
                    continue
                before_distance = float(np.linalg.norm(original[atom_index] - original[a3]))
                after_distance = float(np.linalg.norm(moved[atom_index] - moved[a3]))
                if abs(after_distance - before_distance) > 1e-4:
                    raise ValueError(
                        f"Atom {atom_index} changed its distance to the rotation axis endpoint"
                    )
                radial = original[atom_index] - original[a3]
                radial = radial - np.dot(radial, axis_unit) * axis_unit
                if float(np.linalg.norm(radial)) > 0.05:
                    displacement = float(np.linalg.norm(moved[atom_index] - original[atom_index]))
                    if displacement > 0.05:
                        fragment_moved = True
            if not fragment_moved:
                raise ValueError(
                    f"A {delta} degree mutation of dihedral {tuple(source.atom_indices)} "
                    "did not move the fragment"
                )
            if a4 not in moving:
                raise ValueError(
                    f"The fourth atom of dihedral {tuple(source.atom_indices)} is not in the moving fragment"
                )
            rebuilt = generator.molecule_from_coordinates(child)
            checked = generator.coordinates_from_molecule(rebuilt)
            for stored, actual in zip(child.elements, checked.elements):
                mismatch = _wrapped_degrees(
                    actual.real_values[0], stored.get_actual_value(stored.value)
                )
                if abs(mismatch) > 1e-4:
                    raise ValueError(
                        f"Rebuilt dihedral {tuple(stored.atom_indices)} is {mismatch} "
                        "degrees away from the mutated coordinate"
                    )
            if index in (0, 1) and delta == (30.0 if index == 0 else -30.0):
                rebuilt.properties["dihedral_indices"] = "-".join(
                    str(atom_index) for atom_index in source.atom_indices
                )
                rebuilt.properties["dihedral_delta"] = delta
                selected.append(rebuilt)
    if len(selected) != 2:
        raise ValueError(f"Expected two selected mutants, built {len(selected)}")

    parents = [(copy.deepcopy(template), "initial")]
    offspring = generator.reproduce(parents, 4)
    if len(offspring) != 4 or offspring[0][1] != "initial":
        raise ValueError(f"reproduce did not keep the parent: {[(origin) for _, origin in offspring]}")
    for individual, origin in offspring[1:]:
        if not origin.startswith("mutation-"):
            raise ValueError(f"Expected a mutation origin, got {origin!r}")
        changed = []
        for child_coord, parent_coord in zip(individual.elements, template.elements):
            delta = _wrapped_degrees(
                child_coord.get_actual_value(child_coord.value),
                parent_coord.get_actual_value(parent_coord.value),
            )
            if abs(delta) > 1e-6:
                changed.append((child_coord, delta))
        if len(changed) != 1:
            raise ValueError(f"Mutation changed {len(changed)} dihedrals, expected 1")
        coord, delta = changed[0]
        limit = generator.dihedral_interval
        if coord.bond_type == InternalCoordinateBondType.DOUBLE and generator.db_treatment == "180+step":
            limit += 180.0
        if abs(delta) - limit > 1e-6:
            raise ValueError(f"Mutation step {delta} degrees exceeds the GA interval {limit}")
        rebuilt = generator.molecule_from_coordinates(individual)
        checked = generator.coordinates_from_molecule(rebuilt)
        for stored, actual in zip(individual.elements, checked.elements):
            mismatch = _wrapped_degrees(
                actual.real_values[0], stored.get_actual_value(stored.value)
            )
            if abs(mismatch) > 1e-4:
                raise ValueError(
                    "A reproduce() mutant geometry does not match its dihedral coordinates"
                )
    if not np.allclose(_positions(molecule), original):
        raise ValueError("Dihedral mutation modified the input molecule")
    return selected


@node
async def optimize_mutated_dihedrals(molecule: Molecule, **kwargs) -> MoleculeList:
    """Mutate rotatable dihedrals and relax two of the mutants with DFTB.

    Parameters:
        molecule (Molecule): Structure whose rotatable dihedrals are mutated.

    Called Nodes:
        dftb_list_calculator

    Returns:
        MoleculeList: DFTB-optimized mutated geometries, in the submitted order.
    """
    node_runner = kwargs["node_runner"]
    mutants = mutated_dihedral_molecules(molecule)
    node_runner.info(
        "Dihedral mutations match the Cartesian geometry; "
        f"optimizing {len(mutants)} mutants with dftb_list_calculator"
    )
    batch = MoleculeList()
    submitted = []
    for mutant in mutants:
        copied = Molecule.from_molecule(mutant)
        copied.properties = copy.deepcopy(mutant.properties)
        batch.add_molecule(copied)
        submitted.append(_positions(copied))
    from simstack.models import DataSet

    opts = DftbInput(optimization=True, max_optimization_steps=25)
    calc_result = await dftb_list_calculator(
        batch,
        opts,
        parameters=Parameters(resource="local", in_docker=True, force_rerun=False),
    )
    dataset = getattr(calc_result, "dataset", None)
    if dataset is None and isinstance(calc_result, DataSet):
        dataset = calc_result
    if dataset is None or "results" not in dataset:
        raise ValueError("dftb_list_calculator did not return a results dataset")
    results = dataset["results"]
    if (
        hasattr(results, "data")
        and results.data
        and hasattr(results, "load_to_cache")
        and len(results) == 0
    ):
        await results.load_to_cache(context.db)
    if len(results) != len(mutants):
        raise ValueError(
            f"dftb_list_calculator returned {len(results)} rows for {len(mutants)} molecules"
        )

    optimized = MoleculeList()
    seen = set()
    for _row_name, row in results.items():
        success = row.get("success")
        if success is None:
            raise ValueError("DFTB result row is missing success")
        success_value = success.value if hasattr(success, "value") else success
        if not success_value:
            error = row.get("error")
            message = error.value if hasattr(error, "value") else error
            raise ValueError(f"DFTB optimization failed: {message}")
        source = row.get("arg_molecule")
        if source is None:
            raise ValueError("DFTB result row is missing the input molecule")
        label = source.properties.get("dihedral_indices")
        delta = source.properties.get("dihedral_delta")
        if label is None or delta is None:
            raise ValueError("DFTB result row input is missing the mutation label")
        match = None
        for index, mutant in enumerate(mutants):
            if mutant.properties.get("dihedral_indices") == label and mutant.properties.get("dihedral_delta") == delta:
                match = index
                break
        if match is None or match in seen:
            raise ValueError(f"DFTB result {label} delta {delta} does not match a submitted mutant")
        seen.add(match)
        qm_result = row.get("result_qm_result")
        structure = None if qm_result is None else getattr(qm_result, "final_structure", None)
        if structure is None:
            raise ValueError("DFTB optimization returned no final structure")
        energy = getattr(qm_result, "final_energy", None)
        if energy is None or not np.isfinite(float(energy)):
            raise ValueError("DFTB optimization returned no finite energy")
        if [atom.element for atom in structure.atoms] != [atom.element for atom in mutants[match].atoms]:
            raise ValueError("DFTB optimization changed the atom sequence")
        final_positions = _positions(structure)
        rmsd = float(np.sqrt(np.mean(np.sum((final_positions - submitted[match]) ** 2, axis=1))))
        converged = getattr(qm_result, "optimization_converged", None)
        if rmsd < 1e-3 and converged is not True:
            raise ValueError(
                f"DFTB optimization of dihedral {label} left the geometry unchanged (RMSD {rmsd})"
            )
        result = Molecule.from_molecule(structure)
        result.properties = copy.deepcopy(structure.properties)
        result.properties.update(
            dihedral_indices=label,
            dihedral_delta=delta,
            energy=float(energy) * DFTBEvaluator.HARTREE_TO_KCAL_MOL,
            energy_hartree=float(energy),
            energy_unit="kcal/mol",
            method="dftb",
            optimization_converged=converged,
            optimization_rmsd=rmsd,
        )
        optimized.add_molecule(result)
        node_runner.info(
            f"Optimized dihedral {label} by {delta} degrees: "
            f"E={float(energy):.8f} Ha, RMSD={rmsd:.4f} A, converged={converged}"
        )
    if len(optimized) != len(mutants):
        raise ValueError("DFTB results were not paired with every submitted mutant")
    return optimized
