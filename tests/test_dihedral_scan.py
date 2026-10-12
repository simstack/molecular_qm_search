from pathlib import Path

import numpy as np
from molecular_qm_models import Molecule

from molecular_qm_search.optimization.dihedral_mutation import _positions
from molecular_qm_search.optimization.dihedral_scan import (
    DIHEDRAL_SCAN_DELTAS,
    dihedral_step_lists,
)

TEST1 = Path(__file__).resolve().parents[1] / "data" / "test1.xyz"
EXPECTED_INDICES = (
    ((0, 5, 18, 32), 91.86427498151772),
    ((8, 13, 19, 38), 44.45835381650071),
    ((5, 18, 34, 27), 170.66773732317668),
    ((13, 19, 37, 40), 7.445525044534976),
    ((16, 27, 34, 31), 18.57507376387858),
    ((19, 37, 40, 33), 1.7316481024849901),
)


def test_each_dihedral_is_stepped_from_minus_30_to_plus_30():
    molecule = Molecule.from_xyz(TEST1.read_text())
    original = _positions(molecule)
    lists = dihedral_step_lists(molecule)
    assert len(lists) == len(EXPECTED_INDICES)
    assert np.allclose(_positions(molecule), original)
    for molecule_list, (indices, base) in zip(lists, EXPECTED_INDICES):
        assert len(list(molecule_list)) == len(DIHEDRAL_SCAN_DELTAS)
        label = "-".join(str(atom) for atom in indices)
        for molecule_step, delta in zip(molecule_list, DIHEDRAL_SCAN_DELTAS):
            assert molecule_step.properties["dihedral_indices"] == label
            assert molecule_step.properties["dihedral_delta"] == float(delta)
            angle = molecule_step.properties["dihedral_angle"]
            mismatch = (angle - (base + delta) + 180.0) % 360.0 - 180.0
            assert abs(mismatch) < 1e-3
            if delta == 0:
                assert np.allclose(_positions(molecule_step), original)
