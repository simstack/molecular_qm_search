"""Population construction and genetic operators, independent of energies."""
from __future__ import annotations

import copy
import math
import random

from molecular_qm_models import InternalCoordinatesList, Molecule
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType


def dihedral_rmsd(left: InternalCoordinatesList, right: InternalCoordinatesList) -> float:
    """Circular root-mean-square difference of dihedral angles, in degrees."""
    if len(left.elements) != len(right.elements):
        raise ValueError(
            f"Dihedral lists differ in length ({len(left.elements)} and {len(right.elements)})"
        )
    if not left.elements:
        return 0.0
    squares = []
    for first, second in zip(left.elements, right.elements):
        difference = abs(first.get_actual_value(first.value) - second.get_actual_value(second.value)) % 360
        if difference > 180:
            difference = 360 - difference
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

    def molecule_from_coordinates(self, coordinates: InternalCoordinatesList) -> Molecule:
        result = self._copy_molecule(self.molecule)
        # Coordinate.set() updates the coordinate itself as well as the geometry.
        for coord in copy.deepcopy(coordinates.elements):
            coord.set(result, coord.value)
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

    def reproduce(self, parents, target_size, prune_rms_thresh):
        """Carry parents once and append only new mutations or crossovers.

        Each parent is ``(coordinates, origin, energy)``. Added conformers have
        energy ``None``. A proposal that does not move a dihedral, or whose
        dihedral RMSD to a conformer already kept is below ``prune_rms_thresh``
        degrees, is discarded. After ``target_size`` discarded proposals in a
        row, the returned population is shorter than requested.
        ``last_rejection`` is ``noop`` or ``rmsd``, and ``rejected_rmsd`` is the
        last discarded distance (0 when the proposal did not move).
        """
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
            if not isinstance(parent, tuple) or len(parent) != 3:
                raise ValueError("Parent conformers must carry coordinates, origin, and energy")
        population = list(parents)
        self.rejected_rmsd = None
        self.last_rejection = None
        self.reproduction_stats = {
            "accepted": 0,
            "rejected_rmsd": 0,
            "rejected_noop": 0,
            "closest_rejected_rmsd": None,
        }
        rejected = 0
        n_coords = len(self.coordinates.elements)
        while len(population) < target_size:
            if rejected >= target_size:
                break
            if self.random.random() < self.mutation_rate:
                donor = copy.deepcopy(self.random.choice(parents)[0])
                if n_coords == 0:
                    child, origin, distance = None, None, 0.0
                else:
                    coord = donor.elements[self.random.randrange(n_coords)]
                    delta = self.random.uniform(-self.dihedral_interval, self.dihedral_interval)
                    if (self.db_treatment == "180+step"
                            and coord.bond_type == InternalCoordinateBondType.DOUBLE):
                        delta += 180.0
                    if delta == 0:
                        child, origin, distance = None, None, 0.0
                    else:
                        self._shift(coord, delta)
                        if coord.bond_type is None or not hasattr(coord.bond_type, "value"):
                            raise ValueError(f"Mutated coordinate has no bond type: {coord.bond_type!r}")
                        child, origin = donor, f"mutation-{coord.bond_type.value}"
                        distance = min(dihedral_rmsd(child, kept[0]) for kept in population)
            else:
                if len(parents) < 2:
                    raise ValueError("Crossover requires two parents")
                if n_coords == 0:
                    child, origin, distance = None, None, 0.0
                else:
                    first, second = self.random.sample([individual[0] for individual in parents], 2)
                    point = self.random.randrange(n_coords)
                    child = InternalCoordinatesList(elements=copy.deepcopy(
                        first.elements[:point] + second.elements[point:]
                    ))
                    origin = "crossover"
                    distance = min(dihedral_rmsd(child, kept[0]) for kept in population)
            if child is None or distance < prune_rms_thresh:
                rejected += 1
                if child is None:
                    self.rejected_rmsd = 0.0
                    self.last_rejection = "noop"
                    self.reproduction_stats["rejected_noop"] += 1
                else:
                    self.rejected_rmsd = distance
                    self.last_rejection = "rmsd"
                    self.reproduction_stats["rejected_rmsd"] += 1
                    closest = self.reproduction_stats["closest_rejected_rmsd"]
                    if closest is None or distance < closest:
                        self.reproduction_stats["closest_rejected_rmsd"] = distance
                continue
            rejected = 0
            self.reproduction_stats["accepted"] += 1
            population.append((child, origin, None))
        return population
