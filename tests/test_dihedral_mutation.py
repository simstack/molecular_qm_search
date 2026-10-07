from pathlib import Path

import numpy as np
import pytest
from molecular_qm_models import Molecule
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType

from molecular_qm_search.optimization.dihedral_mutation import (
    mutated_dihedral_molecules,
    optimize_mutated_dihedrals,
)

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
