from pathlib import Path

import numpy as np
import pytest
from molecular_qm_models import Molecule
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType

from molecular_qm_search.optimization.dihedral_mutation import (
    _positions,
    _wrapped_degrees,
    mutated_dihedral_molecules,
    optimize_mutated_dihedrals,
)
from molecular_qm_search.optimization.ga_population import PopulationGenerator

TEST1 = Path(__file__).resolve().parents[1] / "data" / "test1.xyz"

EXPECTED_DIHEDRALS = (
    ((0, 5, 18, 32), 91.86427498151772, 8),
    ((8, 13, 19, 38), 44.45835381650071, 8),
    ((5, 18, 34, 27), 170.66773732317668, 6),
    ((13, 19, 37, 40), 7.445525044534976, 6),
    ((16, 27, 34, 31), 18.57507376387858, 38),
    ((19, 37, 40, 33), 1.7316481024849901, 4),
)


def _test1_molecule():
    if not TEST1.is_file():
        raise ValueError(f"Missing molecule file {TEST1}")
    return Molecule.from_xyz(TEST1.read_text())


def test_test1_rotatable_dihedrals_are_six_single_bonds():
    from molecular_qm_util import get_rotatable_bonds

    molecule = _test1_molecule()
    coordinates = get_rotatable_bonds(
        molecule, match_double_bonds=True, min_value=-180.0, max_value=180.0,
    )
    found = []
    for coord in coordinates.elements:
        found.append((
            tuple(coord.atom_indices),
            coord.real_values[0],
            len(coord.moving_atoms),
            coord.bond_type,
        ))
    assert [(item[0], item[3]) for item in found] == [
        (indices, InternalCoordinateBondType.SINGLE) for indices, _, _ in EXPECTED_DIHEDRALS
    ]
    for actual, expected in zip(found, EXPECTED_DIHEDRALS):
        assert actual[1] == pytest.approx(expected[1])
        assert actual[2] == expected[2]


def _dihedral_angles(individual):
    return [coord.get_actual_value(coord.value) for coord in individual.elements]


def _rmsd(left, right):
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))


def _geometry_matches_dihedrals(generator, individual):
    built = generator.molecule_from_coordinates(individual)
    checked = generator.coordinates_from_molecule(built)
    for stored, actual in zip(individual.elements, checked.elements):
        mismatch = _wrapped_degrees(actual.real_values[0], stored.get_actual_value(stored.value))
        assert abs(mismatch) < 1e-3
    assert [atom.element for atom in built.atoms] == [atom.element for atom in generator.molecule.atoms]
    return built


def _crossover_splice(child_angles, parent_angles):
    count = len(child_angles)
    for first in parent_angles:
        for second in parent_angles:
            for point in range(count):
                mixed = first[:point] + second[point:]
                if any(abs(_wrapped_degrees(got, want)) >= 1e-4 for got, want in zip(child_angles, mixed)):
                    continue
                copied = (
                    all(abs(_wrapped_degrees(got, want)) < 1e-4 for got, want in zip(child_angles, first))
                    or all(abs(_wrapped_degrees(got, want)) < 1e-4 for got, want in zip(child_angles, second))
                )
                return point, copied, first, second
    return None


def test_test1_mutation_and_crossover_generate_new_molecules():
    from molecular_qm_util import get_rotatable_bonds

    molecule = _test1_molecule()
    original = _positions(molecule)
    coordinates = get_rotatable_bonds(
        molecule, match_double_bonds=True, min_value=-180.0, max_value=180.0,
    )
    mutator = PopulationGenerator(
        molecule, coordinates=coordinates, seed=2,
        mutation_rate=1.0, crossover_rate=0.0, dihedral_interval=60.0,
    )
    parents = mutator.generate(5)
    mutated = mutator.reproduce(parents, 9)
    assert [origin for _, origin in mutated[:5]] == ["initial"] * 5
    assert all(origin.startswith("mutation-") for _, origin in mutated[5:])

    parent_angles = [_dihedral_angles(individual) for individual, _ in parents]
    parent_positions = []
    for individual, _ in parents:
        built = _geometry_matches_dihedrals(mutator, individual)
        parent_positions.append(_positions(built))
    assert _rmsd(parent_positions[0], original) < 1e-8
    assert sum(_rmsd(positions, original) > 0.5 for positions in parent_positions[1:]) >= 3

    mutated_positions = []
    for individual, origin in mutated[5:]:
        donor = None
        for angles in parent_angles:
            deltas = [
                _wrapped_degrees(child, parent)
                for child, parent in zip(_dihedral_angles(individual), angles)
            ]
            if sum(abs(delta) > 1e-4 for delta in deltas) == 1 and max(abs(delta) for delta in deltas) <= 60.0 + 1e-6:
                donor = deltas
                break
        assert donor is not None, origin
        built = _geometry_matches_dihedrals(mutator, individual)
        mutated_positions.append(_positions(built))
        assert _rmsd(mutated_positions[-1], original) > 0.05

    crosser = PopulationGenerator(
        molecule, coordinates=coordinates, seed=7,
        mutation_rate=0.0, crossover_rate=1.0, dihedral_interval=60.0,
    )
    crossed = crosser.reproduce(parents, 10)
    assert [origin for _, origin in crossed[:5]] == ["initial"] * 5
    assert [origin for _, origin in crossed[5:]] == ["crossover"] * 5

    novel_crossover = False
    crossover_positions = []
    for individual, _ in crossed[5:]:
        child_angles = _dihedral_angles(individual)
        splice = _crossover_splice(child_angles, parent_angles)
        assert splice is not None
        _point, copied, first, second = splice
        built = _geometry_matches_dihedrals(crosser, individual)
        positions = _positions(built)
        crossover_positions.append(positions)
        if copied:
            continue
        first_positions = parent_positions[parent_angles.index(first)]
        second_positions = parent_positions[parent_angles.index(second)]
        if _rmsd(positions, first_positions) > 0.05 and _rmsd(positions, second_positions) > 0.05:
            novel_crossover = True
    assert novel_crossover

    distinct = [original, *parent_positions[1:], *mutated_positions, *crossover_positions]
    separated = 0
    for index, left in enumerate(distinct):
        for right in distinct[index + 1:]:
            if _rmsd(left, right) > 0.05:
                separated += 1
    assert separated >= 10
    assert np.allclose(_positions(molecule), original)


def test_test1_ga_mutation_steps_change_geometry_by_the_requested_angle():
    molecule = _test1_molecule()
    before = np.array([[atom.x, atom.y, atom.z] for atom in molecule.atoms])
    mutants = mutated_dihedral_molecules(molecule)
    after = np.array([[atom.x, atom.y, atom.z] for atom in molecule.atoms])
    assert np.allclose(before, after)
    assert [mutant.properties["dihedral_delta"] for mutant in mutants] == [30.0, -30.0]
    assert [mutant.properties["dihedral_indices"] for mutant in mutants] == [
        "0-5-18-32",
        "8-13-19-38",
    ]
    for mutant in mutants:
        positions = np.array([[atom.x, atom.y, atom.z] for atom in mutant.atoms])
        rmsd = float(np.sqrt(np.mean(np.sum((positions - before) ** 2, axis=1))))
        assert rmsd > 0.1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dftb_optimizes_mutated_test1_dihedrals_in_one_container():
    from simstack.core.context import context
    from simstack.models import Parameters
    from simstack.util.project_root_finder import find_project_root

    project_root = find_project_root()
    if not (Path(project_root) / "simstack.toml").exists():
        pytest.skip(
            "simstack.toml is not in the project root; set SIMSTACK_PROJECT_ROOT "
            "to the project whose local runner is already running"
        )
    await context.initialize()
    optimized = await optimize_mutated_dihedrals(
        _test1_molecule(),
        parameters=Parameters(force_rerun=True),
    )
    assert len(optimized) == 2
    labels = []
    for molecule in optimized:
        assert molecule.properties["method"] == "dftb"
        assert molecule.properties["energy_unit"] == "kcal/mol"
        assert molecule.properties["energy"] == pytest.approx(
            molecule.properties["energy_hartree"] * 627.5094740631
        )
        assert molecule.properties["optimization_rmsd"] > 1e-3 or molecule.properties["optimization_converged"] is True
        labels.append((molecule.properties["dihedral_indices"], molecule.properties["dihedral_delta"]))
    assert sorted(labels) == [("0-5-18-32", 30.0), ("8-13-19-38", -30.0)]
