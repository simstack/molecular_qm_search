"""Population construction and genetic operators, independent of energies."""
from __future__ import annotations

import copy
import random

from molecular_qm_models import InternalCoordinatesList, Molecule
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType


class PopulationGenerator:
    def __init__(self, molecule: Molecule, *, coordinates=None, seed=1,
                 mutation_rate=0.2, crossover_rate=0.5, dihedral_interval=30.0,
                 db_treatment="180+step", match_double_bonds=True,
                 rotatable_bond_min=-180.0, rotatable_bond_max=180.0):
        if db_treatment not in {"ignore", "180+step", "treat-as-single"}:
            raise ValueError(f"Unknown double-bond treatment: {db_treatment}")
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
        self.crossover_rate = crossover_rate
        self.dihedral_interval = dihedral_interval
        self.db_treatment = db_treatment

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

    def generate(self, size: int) -> list[tuple[InternalCoordinatesList, str]]:
        if size < 1:
            raise ValueError("Population size must be positive")
        population = [(copy.deepcopy(self.coordinates), "initial")]
        for _ in range(size - 1):
            individual = copy.deepcopy(self.coordinates)
            for coord in individual.elements:
                self._shift(coord, self.random.uniform(-180, 180))
            population.append((individual, "initial"))
        return population

    def reproduce(self, parents, target_size):
        if not parents:
            raise ValueError("Cannot reproduce an empty population")
        population = copy.deepcopy(parents[:target_size])
        n_coords = len(self.coordinates.elements)
        while len(population) < target_size:
            if n_coords and self.random.random() < self.crossover_rate and len(parents) >= 2:
                first, second = self.random.sample([ind for ind, _ in parents], 2)
                point = self.random.randrange(n_coords)
                child = InternalCoordinatesList(elements=copy.deepcopy(
                    first.elements[:point] + second.elements[point:]
                ))
                origin = "crossover"
            else:
                child = copy.deepcopy(self.random.choice(parents)[0])
                origin = "copy"
                if n_coords and self.random.random() < self.mutation_rate:
                    coord = child.elements[self.random.randrange(n_coords)]
                    delta = self.random.uniform(-self.dihedral_interval, self.dihedral_interval)
                    if (self.db_treatment == "180+step"
                            and coord.bond_type == InternalCoordinateBondType.DOUBLE):
                        delta += 180.0
                    self._shift(coord, delta)
                    if coord.bond_type is None or not hasattr(coord.bond_type, "value"):
                        raise ValueError(f"Mutated coordinate has no bond type: {coord.bond_type!r}")
                    origin = f"mutation-{coord.bond_type.value}"
            population.append((child, origin))
        return population
