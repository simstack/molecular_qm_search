"""Population construction and genetic operators, independent of energies."""
from __future__ import annotations

import copy
import math
import random

from molecular_qm_models import InternalCoordinatesList, Molecule
from molecular_qm_models.internal_coordinates import (
    InternalCoordinateBase,
    InternalCoordinateBondType,
)


def dihedral_rmsd(
        left: InternalCoordinatesList,
        right: InternalCoordinatesList,
        symmetries=None,
) -> float:
    """Circular root-mean-square difference of dihedral angles, in degrees.

    ``symmetries`` has one entry per dihedral. ``None`` is an ordinary torsion.
    A number is the angle, in degrees, between the two hydrogens of a terminal
    group such as NH2. Those hydrogens are indistinguishable, so an angle and
    the same angle plus or minus that shift describe one conformation.
    """
    if len(left.elements) != len(right.elements):
        raise ValueError(
            f"Dihedral lists differ in length ({len(left.elements)} and {len(right.elements)})"
        )
    if symmetries is not None and len(symmetries) != len(left.elements):
        raise ValueError(
            f"Dihedral symmetries ({len(symmetries)}) do not match the "
            f"{len(left.elements)} dihedrals"
        )
    if not left.elements:
        return 0.0
    squares = []
    for index, (first, second) in enumerate(zip(left.elements, right.elements)):
        start = float(first.get_actual_value(first.value))
        finish = float(second.get_actual_value(second.value))
        difference = abs(start - finish) % 360
        if difference > 180:
            difference = 360 - difference
        if symmetries is not None and symmetries[index] is not None:
            if not math.isfinite(float(symmetries[index])):
                raise ValueError(f"Dihedral {index} has a non-finite symmetry shift")
            # The shift is not exactly 180, so the two hydrogens are not inverses
            # of each other. Both directions map one hydrogen's angle onto the other.
            for sign in (1.0, -1.0):
                shifted = abs(start - (finish - sign * float(symmetries[index]))) % 360
                if shifted > 180:
                    shifted = 360 - shifted
                difference = min(difference, shifted)
        squares.append(difference * difference)
    return math.sqrt(sum(squares) / len(squares))


class PopulationGenerator:
    def __init__(self, molecule: Molecule, *, coordinates=None, seed=1,
                 mutation_rate=0.2, dihedral_interval=30.0,
                 db_treatment="180+step", match_double_bonds=True,
                 rotatable_bond_min=-180.0, rotatable_bond_max=180.0):
        if db_treatment not in {"ignore", "180+step", "treat-as-single"}:
            raise ValueError(f"Unknown double-bond treatment: {db_treatment}")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must lie in [0, 1]")
        if rotatable_bond_max <= rotatable_bond_min:
            raise ValueError(
                f"rotatable_bond_max ({rotatable_bond_max}) must be greater than "
                f"rotatable_bond_min ({rotatable_bond_min})"
            )
        self.molecule = self._copy_molecule(molecule)
        if coordinates is None:
            from molecular_qm_util import get_rotatable_bonds

            coordinates = get_rotatable_bonds(
                molecule,
                match_double_bonds=match_double_bonds,
                min_value=rotatable_bond_min,
                max_value=rotatable_bond_max,
            )
        self.coordinates = copy.deepcopy(coordinates)
        if db_treatment == "ignore":
            self.coordinates.elements = [c for c in self.coordinates.elements
                                         if c.bond_type != InternalCoordinateBondType.DOUBLE]
        elif db_treatment == "treat-as-single":
            for coord in self.coordinates.elements:
                coord.bond_type = InternalCoordinateBondType.SINGLE
        for coord in self.coordinates.elements:
            coord.compute(self.molecule)
        self.random = random.Random(seed)
        self.mutation_rate = mutation_rate
        self.dihedral_interval = dihedral_interval
        self.db_treatment = db_treatment
        self.rejected_rmsd = None
        self.last_rejection = None
        self.reproduction_stats = None

    def _chosen_fragment(self, coord) -> tuple[list[int], bool]:
        """Smaller side of the bond, and whether that side is the atom-2 side.

        Equal sizes use the atom-2 side. ``InternalDihedralCoordinate.set``
        realizes the requested angle on that side and turns the atom-4 side
        backwards.
        """
        atom1, atom2, atom3, atom4 = coord.atom_indices
        side_2 = InternalCoordinateBase.get_moving_atoms(
            self.molecule, (atom3, atom2), collision_set=[atom4]
        )
        side_4 = InternalCoordinateBase.get_moving_atoms(
            self.molecule, (atom2, atom3), collision_set=[atom1]
        )
        use_atom2_side = bool(side_2) and len(side_2) <= len(side_4)
        moving = side_2 if use_atom2_side else side_4
        if not moving:
            raise ValueError(
                f"Dihedral {tuple(coord.atom_indices)} has no rotatable fragment"
            )
        return moving, use_atom2_side

    def _apply_dihedral(self, coord, molecule: Molecule) -> list[int]:
        """Rotate the smaller side until the geometry shows ``coord``'s angle.

        The atom-4 side is used only when it is smaller. Its target is reflected
        so the backwards rotation in ``set`` still hits the stored angle.
        """
        moving, use_atom2_side = self._chosen_fragment(coord)
        target = float(coord.get_actual_value(coord.value))
        if not math.isfinite(target):
            raise ValueError(
                f"Dihedral {tuple(coord.atom_indices)} has a non-finite angle {target}"
            )
        low, high = float(coord.min_values[0]), float(coord.max_values[0])
        span = high - low
        if span <= 0:
            raise ValueError(
                f"Dihedral {tuple(coord.atom_indices)} bounds must have positive width"
            )
        current = copy.deepcopy(coord)
        current.moving_atoms = list(moving)
        if use_atom2_side:
            value = coord.value
        else:
            probe = copy.deepcopy(coord)
            probe.compute(molecule)
            current_angle = float(probe.real_values[0])
            if not math.isfinite(current_angle):
                raise ValueError(
                    f"Dihedral {tuple(coord.atom_indices)} has a non-finite geometry angle"
                )
            reflected = (2.0 * current_angle - target - low) % span + low
            value = (reflected - low) / span
        current.set(molecule, value)
        measured = float(current.real_values[0])
        mismatch = (measured - target + 180.0) % 360.0 - 180.0
        if abs(mismatch) > 1e-4:
            raise ValueError(
                f"Dihedral {tuple(coord.atom_indices)} measured {measured} degrees, "
                f"expected {target}"
            )
        return list(moving)

    def molecule_from_coordinates(self, coordinates: InternalCoordinatesList) -> Molecule:
        result = self._copy_molecule(self.molecule)
        # Larger fragments first, so a later smaller torsion can set its own angle
        # without being overwritten by a rotation of the rest of the molecule.
        ordered = sorted(
            coordinates.elements,
            key=lambda coord: len(self._chosen_fragment(coord)[0]),
            reverse=True,
        )
        for coord in ordered:
            self._apply_dihedral(coord, result)
        for key in ("energy", "energy_hartree", "energy_unit", "method", "rank_id",
                    "optimization_converged"):
            result.properties.pop(key, None)
        return result

    @staticmethod
    def _copy_molecule(molecule: Molecule) -> Molecule:
        # Reconstruct atoms to retain ODMantic's change-tracking fields.
        result = Molecule.from_molecule(molecule)
        result.properties = copy.deepcopy(molecule.properties)
        for atom, source in zip(result.atoms, molecule.atoms):
            atom.properties = copy.deepcopy(source.properties)
        return result

    def coordinates_from_molecule(self, molecule: Molecule) -> InternalCoordinatesList:
        result = copy.deepcopy(self.coordinates)
        for coord in result.elements:
            coord.compute(molecule)
        return result

    @staticmethod
    def _shift(coord, delta):
        low, high = coord.min_values[0], coord.max_values[0]
        span = high - low
        if span <= 0:
            raise ValueError("Coordinate bounds must have positive width")
        angle = (coord.get_actual_value(coord.value) + delta - low) % span + low
        coord.value = (angle - low) / span
        coord.real_values = [angle]

    def generate(self, size: int) -> list[tuple[InternalCoordinatesList, str, None]]:
        if size < 1:
            raise ValueError("Population size must be positive")
        population = [(copy.deepcopy(self.coordinates), "initial", None)]
        for _ in range(size - 1):
            individual = copy.deepcopy(self.coordinates)
            for coord in individual.elements:
                self._shift(coord, self.random.uniform(-180, 180))
            population.append((individual, "initial", None))
        return population

    def reproduce(self, parents, target_size, prune_rms_thresh, history, *, neighbor_edges=None):
        """Carry parents once and append new mutations or crossovers.

        Each parent is ``(coordinates, origin, energy, history_index)``.
        ``history`` is the list of operations that created every configuration.
        A mutation appends ``M`` with the ancestor index, dihedral index, and
        signed angle in degrees. A crossover appends ``X`` with both ancestor
        indices and the dihedral index where the second ancestor begins.
        Added conformers have energy ``None``. A proposal that does not move a
        dihedral is discarded and is not numbered. Dihedral RMSD is not applied
        here: the caller scores the children and then keeps the lower-energy
        conformer of each neighborhood.
        After ``target_size - len(parents)`` discarded proposals in a row, the
        returned population is shorter than requested. ``last_rejection`` is
        ``noop`` when that happens, and ``rejected_rmsd`` stays unset.
        ``neighbor_edges`` restricts crossover to those pairs of parent indexes.
        An empty edge list raises ``ValueError`` when a crossover is required.
        """
        if not isinstance(history, list):
            raise ValueError("GA history must be a list")
        if not parents:
            raise ValueError("Cannot reproduce an empty population")
        if target_size < 1:
            raise ValueError("Population size must be positive")
        if not math.isfinite(prune_rms_thresh) or prune_rms_thresh <= 0:
            raise ValueError(
                f"prune_rms_thresh must be a positive finite dihedral RMSD in degrees, "
                f"got {prune_rms_thresh}"
            )
        if len(parents) > target_size:
            raise ValueError(
                f"Cannot carry {len(parents)} parents into a population of {target_size}"
            )
        for parent in parents:
            if not isinstance(parent, tuple) or len(parent) != 4:
                raise ValueError(
                    "Parent conformers must carry coordinates, origin, energy, and history index"
                )
            ancestor = parent[3]
            if not isinstance(ancestor, int) or not 0 <= ancestor < len(history):
                raise ValueError(f"Configuration {ancestor} is not in the GA history")
        population = list(parents)
        self.rejected_rmsd = None
        self.last_rejection = None
        self.reproduction_stats = {
            "accepted": 0,
            "rejected_rmsd": 0,
            "rejected_noop": 0,
            "closest_rejected_rmsd": None,
        }
        resolved_edges = None
        if neighbor_edges is not None:
            resolved_edges = []
            for edge in neighbor_edges:
                if not isinstance(edge, tuple) or len(edge) != 2:
                    raise ValueError(
                        f"Neighbor edge must be a pair of population indexes, got {edge!r}"
                    )
                left, right = edge
                if (
                    isinstance(left, bool) or isinstance(right, bool)
                    or not isinstance(left, int) or not isinstance(right, int)
                    or left == right
                    or not 0 <= left < len(parents)
                    or not 0 <= right < len(parents)
                ):
                    raise ValueError(
                        f"Neighbor edge {edge!r} is not a pair of distinct population indexes"
                    )
                resolved_edges.append((left, right))
        child_quota = target_size - len(parents)
        rejected = 0
        n_coords = len(self.coordinates.elements)
        while len(population) < target_size:
            if rejected >= child_quota:
                break
            if len(parents) < 2 and self.mutation_rate == 0:
                raise ValueError("Crossover requires two parents")
            operation = None
            if self.random.random() < self.mutation_rate or len(parents) < 2:
                parent = self.random.choice(parents)
                donor = copy.deepcopy(parent[0])
                if n_coords == 0:
                    child, origin = None, None
                else:
                    dihedral_index = self.random.randrange(n_coords)
                    coord = donor.elements[dihedral_index]
                    delta = self.random.uniform(-self.dihedral_interval, self.dihedral_interval)
                    if (self.db_treatment == "180+step"
                            and coord.bond_type == InternalCoordinateBondType.DOUBLE):
                        delta += 180.0
                    if delta == 0:
                        child, origin = None, None
                    else:
                        self._shift(coord, delta)
                        if coord.bond_type is None or not hasattr(coord.bond_type, "value"):
                            raise ValueError(f"Mutated coordinate has no bond type: {coord.bond_type!r}")
                        child, origin = donor, f"mutation-{coord.bond_type.value}"
                        operation = {
                            "op": "M",
                            "ancestor": parent[3],
                            "dihedral": dihedral_index,
                            "angle": float(delta),
                        }
            else:
                if n_coords == 0:
                    child, origin = None, None
                else:
                    if resolved_edges is not None:
                        if not resolved_edges:
                            raise ValueError(
                                "Crossover requires a neighbor edge, and the weighted "
                                "neighbor graph has none"
                            )
                        left, right = self.random.choice(resolved_edges)
                        first, second = parents[left], parents[right]
                    else:
                        first, second = self.random.sample(parents, 2)
                    point = self.random.randrange(n_coords)
                    child = InternalCoordinatesList(elements=copy.deepcopy(
                        first[0].elements[:point] + second[0].elements[point:]
                    ))
                    origin = "crossover"
                    operation = {
                        "op": "X",
                        "a1": first[3],
                        "a2": second[3],
                        "cut": point,
                    }
            if child is None:
                rejected += 1
                self.rejected_rmsd = 0.0
                self.last_rejection = "noop"
                self.reproduction_stats["rejected_noop"] += 1
                continue
            if operation is None:
                raise ValueError("Created conformer has no history operation")
            rejected = 0
            self.reproduction_stats["accepted"] += 1
            history_index = len(history)
            operation["index"] = history_index
            history.append(operation)
            population.append((child, origin, None, history_index))
        return population
