"""Inspection geometries that step one rotatable dihedral at a time."""
import copy

from molecular_qm_models import Molecule, MoleculeList
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult

from molecular_qm_search.optimization.ga_population import PopulationGenerator

DIHEDRAL_SCAN_DELTAS = (-30, -20, -10, 0, 10, 20, 30)


def dihedral_step_lists(molecule: Molecule) -> list[MoleculeList]:
    """One list per rotatable dihedral, stepped from -30 to +30 degrees.

    Each geometry starts from the input coordinates. Only that dihedral is
    rotated, in steps of 10 degrees. Every other rotatable dihedral must keep
    its input angle.
    """
    from molecular_qm_util import get_rotatable_bonds

    if molecule is None or not molecule.atoms:
        raise ValueError("Dihedral scan requires a molecule with atoms")
    coordinates = get_rotatable_bonds(
        molecule, match_double_bonds=True, min_value=-180.0, max_value=180.0,
    )
    if not coordinates.elements:
        raise ValueError("No rotatable dihedrals were found")
    generator = PopulationGenerator(
        molecule,
        coordinates=coordinates,
        seed=1,
        mutation_rate=1.0,
        dihedral_interval=10.0,
        db_treatment="180+step",
        match_double_bonds=True,
        rotatable_bond_min=-180.0,
        rotatable_bond_max=180.0,
    )
    template = generator.coordinates
    lists = []
    for index, source in enumerate(template.elements):
        if len(source.atom_indices) != 4:
            raise ValueError(f"Dihedral atom indices must have length 4, got {source.atom_indices}")
        indices = "-".join(str(atom) for atom in source.atom_indices)
        base = float(source.get_actual_value(source.value))
        molecule_list = MoleculeList()
        for delta in DIHEDRAL_SCAN_DELTAS:
            child = copy.deepcopy(source)
            if delta != 0:
                generator._shift(child, float(delta))
            target = float(child.get_actual_value(child.value))
            moved = (target - base + 180.0) % 360.0 - 180.0
            if abs(moved - delta) > 1e-6:
                raise ValueError(
                    f"Dihedral {indices} stored angle moved by {moved} degrees, expected {delta}"
                )
            isolated = generator._copy_molecule(molecule)
            generator._apply_dihedral(child, isolated)
            built = generator.coordinates_from_molecule(isolated)
            if len(built.elements) != len(template.elements):
                raise ValueError("Dihedral scan changed the number of rotatable dihedrals")
            for other_index, (original, actual) in enumerate(zip(template.elements, built.elements)):
                expected = target if other_index == index else float(
                    original.get_actual_value(original.value)
                )
                measured = float(actual.real_values[0])
                mismatch = (measured - expected + 180.0) % 360.0 - 180.0
                if abs(mismatch) > 1e-3:
                    raise ValueError(
                        f"Stepping dihedral {indices} by {delta} degrees left dihedral "
                        f"{tuple(original.atom_indices)} {mismatch} degrees from {expected}"
                    )
            isolated.properties["dihedral_indices"] = indices
            isolated.properties["dihedral_delta"] = float(delta)
            isolated.properties["dihedral_angle"] = target
            isolated.properties["label"] = f"{indices} {delta:+.0f}"
            molecule_list.add_molecule(isolated)
        lists.append(molecule_list)
    return lists


@node
async def scan_dihedral_steps(molecule: Molecule, **kwargs) -> SimstackResult:
    """Step each rotatable dihedral from -30 to +30 degrees for inspection.

    Parameters:
        molecule (Molecule): Structure whose rotatable dihedrals are scanned.

    Raises:
        ValueError: the molecule has no atoms or no rotatable dihedrals, or a step
        does not appear in the Cartesian geometry.

    SimstackResult:
        dihedral_scan_0 (MoleculeList): Seven geometries of the first rotatable dihedral, ordered from -30 to +30 degrees in steps of 10. Further dihedrals are dihedral_scan_1, dihedral_scan_2, and so on.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("scan_dihedral_steps requires node_runner")
    lists = dihedral_step_lists(molecule)
    for index, molecule_list in enumerate(lists):
        setattr(node_runner, f"dihedral_scan_{index}", molecule_list)
    node_runner.info(
        f"Dihedral scan built {len(lists)} lists with steps {list(DIHEDRAL_SCAN_DELTAS)}"
    )
    return node_runner.succeed()
