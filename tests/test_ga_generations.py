import copy
import logging

import numpy as np
import pytest
from molecular_qm_models import Atom, InternalCoordinatesList, InternalDihedralCoordinate, Molecule

from molecular_qm_search.optimization.dihedral_mutation import _positions
from molecular_qm_search.optimization.ga_generations import (
    GA_INSPECTION_POP_SIZE,
    ga_generation_lists,
)


class Evaluator:
    """A geometry-changing backend with an energy independent of the torsion."""

    def score(self, molecules):
        results = [Molecule.from_molecule(molecule) for molecule in molecules]
        for molecule in results:
            molecule.properties["energy"] = 5.0
        return results

    def optimize(self, molecules, *, max_iters):
        results = [Molecule.from_molecule(molecule) for molecule in molecules]
        for molecule in results:
            molecule.properties.update(energy=-2.0, energy_unit="kcal/mol", method="test")
        return results


@pytest.fixture
def molecule():
    return Molecule(atoms=[
        Atom(element="C", x=0, y=1, z=0),
        Atom(element="C", x=0, y=0, z=0),
        Atom(element="C", x=1.5, y=0, z=0),
        Atom(element="C", x=1.5, y=1, z=1),
    ])


@pytest.fixture
def coordinates(molecule):
    coord = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180, 180)
    coord.moving_atoms = [2, 3]
    coord.compute(molecule)
    return InternalCoordinatesList(elements=[coord])


def test_three_generations_start_from_the_input(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = _positions(molecule)
    lists = ga_generation_lists(
        molecule,
        node_runner=logging.getLogger("ga-generation-test"),
        evaluator=Evaluator(),
        coordinates=copy.deepcopy(coordinates),
    )
    assert len(lists) == 3
    assert np.allclose(_positions(molecule), original)
    first_generation = list(lists[0])
    assert len(first_generation) == GA_INSPECTION_POP_SIZE
    assert np.allclose(_positions(first_generation[0]), original)
    assert first_generation[0].properties["generation"] == 0
    assert first_generation[0].properties["origin"] == "initial"
    for generation, molecule_list in enumerate(lists):
        conformers = list(molecule_list)
        assert 1 <= len(conformers) <= GA_INSPECTION_POP_SIZE
        assert [item.properties["generation"] for item in conformers] == [generation] * len(conformers)
