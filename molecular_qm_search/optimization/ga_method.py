from __future__ import annotations
import asyncio
import copy
import numpy as np
import time
import pickle
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

from molecular_qm_models import Molecule, MoleculeList, InternalCoordinatesList
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType
from simstack.core.node_runner import NodeRunner
from simstack.models.charts_artifact import (
    AGChartAxisConfig,
    AGChartLegendConfig,
    AGChartTitleConfig,
    AGColumnSeriesConfig,
    AGLineSeriesConfig,
    AGScatterSeriesConfig,
    ChartArtifactModel,
)
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

from .ga_population import PopulationGenerator, dihedral_rmsd
from molecular_qm_search.optimization.lib.ga_evaluation import MoleculeEvaluator, make_evaluator, validate_results
from molecular_qm_search.optimization.lib.ga_smart_optimizer import SmartOptimizer
from molecular_qm_search.optimization.models.ga_models import GAOptimizationMethod


def plot_conformer_diversity(conformers: MoleculeList, output_prefix: str = "") -> None:
    """Helper to plot conformer diversity if needed."""
    pass


_COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
}
_VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "S": 1.80, "P": 1.80, "Cl": 1.75,
    "B": 1.92, "Si": 2.10,
}
# Non-bonded pairs closer than this fraction of the van der Waals sum are overlaps.
# 0.6 puts an H-H limit at 1.44 A, so 0.982 A fails and a normal 1.7 A contact does not.
_OVERLAP_VDW_FRACTION = 0.6


def _to_molecule_list(molecules) -> MoleculeList:
    """Collect an iterable of qm_models ``Molecule`` objects into a ``MoleculeList``."""
    mol_list = MoleculeList()
    for mol in molecules:
        mol_list.add_molecule(mol)
    return mol_list


class BaseGA:
    def __init__(
            self,
            initial_mol: Molecule,
            node_runner: NodeRunner,
            num_confs: int = 50,
            pop_size: int = 100,
            generations: int = 50,
            mutation_rate: float = 0.2,
            dihedral_interval: float = 30.0,
            prune_rms_thresh: float = 20.0,
            seed: int = 1,
            max_iters: int = 500,
            parallel_children: int = 0,
            profile: bool = False,
            smart_opt: bool = False,
            restart: bool = False,
            db_treatment: str = "180+step",
            match_double_bonds: bool = True,
            rotatable_bond_min: float = -180.0,
            rotatable_bond_max: float = 180.0,
            evaluator: Optional[MoleculeEvaluator] = None,
            coordinates: Optional[InternalCoordinatesList] = None,
            optimization_method: GAOptimizationMethod = GAOptimizationMethod.RDKIT_MMFF,
            backend_options=None,
            artifact_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.num_confs = num_confs
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.dihedral_interval = dihedral_interval
        self.prune_rms_thresh = prune_rms_thresh
        self.seed = seed
        self.max_iters = max_iters
        self.parallel_children = parallel_children
        self.profile = profile
        self.smart_opt = smart_opt
        self.restart = restart
        self.smart_optimizer = SmartOptimizer(active=smart_opt)
        self.node_runner = node_runner
        self.best_energy_seen = float('inf')
        self.best_stored: MoleculeList = MoleculeList()

        # Initial structure input (qm_models Molecule)
        self.initial_mol: Molecule = initial_mol
        self.template_coords: InternalCoordinatesList = InternalCoordinatesList()
        self.dihedrals: List[Tuple[int, int, int, int]] = []
        self.dihedral_types: List[str] = []  # 'SB' (single bond) or 'DB' (double bond)
        self.db_treatment = db_treatment
        self.match_double_bonds = match_double_bonds
        self.rotatable_bond_min = rotatable_bond_min
        self.rotatable_bond_max = rotatable_bond_max
        self.timing: Dict[str, float] = {}
        self.energy_history: List[Dict[str, float]] = []
        self._energy_plot_start: Optional[int] = None
        self.artifact_loop = artifact_loop
        # [kept after selection, produced] for each population-change label.
        # Each conformer is counted once, on the generation that creates it.
        self.success_stats: Dict[str, List[int]] = {}
        self._counted_history: set[int] = set()

        if pop_size < 1 or num_confs < 1 or generations < 0:
            raise ValueError("Population and conformer counts must be positive; generations cannot be negative")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must lie in [0, 1]")
        if not np.isfinite(prune_rms_thresh) or prune_rms_thresh <= 0:
            raise ValueError(
                f"prune_rms_thresh must be a positive finite dihedral RMSD in degrees, "
                f"got {prune_rms_thresh}"
            )
        if max_iters < 0 or parallel_children < 0 or dihedral_interval < 0:
            raise ValueError("Iteration, parallel children, and dihedral-step values cannot be negative")
        if rotatable_bond_max <= rotatable_bond_min:
            raise ValueError(
                f"rotatable_bond_max ({rotatable_bond_max}) must be greater than "
                f"rotatable_bond_min ({rotatable_bond_min})"
            )
        self._bonded_pairs = None
        self._dihedral_symmetries = None
        self._history: List[dict] = []
        self._conformation_rows: List[Dict[str, Union[int, float, str]]] = []
        self.evaluator = evaluator if evaluator is not None else make_evaluator(
            optimization_method, parallel_children=parallel_children,
            backend_options=backend_options,
        )
        self.input_coordinates = coordinates
        self.population_generator = None
        self.evaluated_molecules: List[Molecule] = []
        self._dihedral_histogram_names: List[str] = []

    def molecule_from_coordinates(self, coords: InternalCoordinatesList) -> Molecule:
        return self.population_generator.molecule_from_coordinates(coords)

    def coordinates_from_molecule(self, mol: Molecule) -> InternalCoordinatesList:
        return self.population_generator.coordinates_from_molecule(mol)

    def evaluate_molecules(self, molecules: List[Molecule], *, optimize=False,
                           max_iters=None) -> List[Molecule]:
        if not molecules:
            return []
        if optimize:
            results = self.evaluator.optimize(
                molecules, max_iters=self.max_iters if max_iters is None else max_iters
            )
        else:
            results = self.evaluator.score(molecules)
        return validate_results(molecules, results)

    def save_state(self, population, gen: int):
        state = {
            "population": population,
            "gen": gen,
            "best_stored": self.best_stored,
            "success_stats": self.success_stats,
            "counted_history": sorted(self._counted_history),
            "energy_history": self.energy_history,
            "best_energy_seen": self.best_energy_seen,
            "random_state": self.population_generator.random.getstate(),
            "history": self._history,
            "minimized_start": PopulationGenerator._copy_molecule(
                self.population_generator.molecule
            ),
        }
        if hasattr(self, "rmsd_history"):
            state["rmsd_history"] = self.rmsd_history
        if hasattr(self, "relaxed"):
            state["relaxed"] = [self.relaxed.get(id(coords)) for coords, _, _, _ in population]

        with open("ga_state.pkl", "wb") as f:
            pickle.dump(state, f)
        self._report(f"Saved GA state of generation {gen}.")

    def load_state(self):
        if not Path("ga_state.pkl").exists():
            return None
        try:
            with open("ga_state.pkl", "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            self.node_runner.error(f"Failed to load GA state: {e}")
            return None
        if "random_state" in state:
            self.population_generator.random.setstate(state["random_state"])
        if "energy_history" not in state:
            raise ValueError("GA checkpoint has no energy_history")
        if "success_stats" not in state:
            raise ValueError("GA checkpoint has no success_stats")
        if "counted_history" not in state or not isinstance(state["counted_history"], list):
            raise ValueError("GA checkpoint has no counted history indexes")
        population = state.get("population")
        if not population or any(
            not isinstance(item, tuple) or len(item) != 4 or item[2] is None for item in population
        ):
            raise ValueError("GA checkpoint population has no energies")
        if "history" not in state or not isinstance(state["history"], list):
            raise ValueError("GA checkpoint has no history")
        self._history = state["history"]
        for item in population:
            history_index = item[3]
            if not isinstance(history_index, int) or not 0 <= history_index < len(self._history):
                raise ValueError(f"Configuration {history_index} is not in the GA history")
        self.best_stored = state.get("best_stored", MoleculeList())
        self.success_stats = state["success_stats"]
        self._counted_history = set(state["counted_history"])
        self.energy_history = state["energy_history"]
        self.best_energy_seen = state.get("best_energy_seen", float('inf'))
        if hasattr(self, "rmsd_history") and "rmsd_history" in state:
            self.rmsd_history = state["rmsd_history"]
        if hasattr(self, "relaxed"):
            relaxed = state.get("relaxed")
            if relaxed is None or len(relaxed) != len(population):
                raise ValueError("GA checkpoint is missing relaxed geometries")
            self.relaxed = {}
            for (coords, _, _, _), molecule in zip(population, relaxed):
                if molecule is not None:
                    self.relaxed[id(coords)] = molecule
        minimized_start = state.get("minimized_start")
        if minimized_start is None:
            raise ValueError("GA checkpoint has no minimized starting geometry")
        self._replace_search_geometry(minimized_start)
        self._report(f"Loaded GA state from generation {state['gen']}.")
        return population, state["gen"]

    def setup(self):
        """Prepare internal coordinates without scoring or optimizing geometry."""
        t0 = time.perf_counter()
        if self.initial_mol is None:
            raise ValueError("Provide an initial Molecule")
        self.population_generator = PopulationGenerator(
            self.initial_mol, coordinates=self.input_coordinates, seed=self.seed,
            mutation_rate=self.mutation_rate,
            dihedral_interval=self.dihedral_interval, db_treatment=self.db_treatment,
            match_double_bonds=self.match_double_bonds,
            rotatable_bond_min=self.rotatable_bond_min,
            rotatable_bond_max=self.rotatable_bond_max,
        )
        self.template_coords = self.population_generator.coordinates
        self.dihedrals = [tuple(c.atom_indices) for c in self.template_coords.elements]
        self.dihedral_types = [
            'DB' if c.bond_type == InternalCoordinateBondType.DOUBLE else 'SB'
            for c in self.template_coords.elements
        ]
        self._bonded_pairs = self._bonds_of(self.initial_mol)
        self._set_dihedral_symmetries()
        bind_connectivity = getattr(self.evaluator, "bind_connectivity", None)
        if bind_connectivity is not None:
            bind_connectivity(self.initial_mol)
        self.timing["Setup"] = time.perf_counter() - t0

    def _set_dihedral_symmetries(self):
        """Record the angle between equivalent terminal hydrogens on the current geometry."""
        self._dihedral_symmetries = []
        for coord in self.template_coords.elements:
            atom1, atom2, atom3, atom4 = (int(index) for index in coord.atom_indices)
            neighbors = [
                other for other in range(len(self.initial_mol.atoms))
                if other != atom3 and (min(atom3, other), max(atom3, other)) in self._bonded_pairs
            ]
            other_hydrogens = [
                other for other in neighbors
                if other != atom4 and self.initial_mol.atoms[other].element == "H"
            ]
            terminal_pair = (
                self.initial_mol.atoms[atom4].element == "H"
                and len(other_hydrogens) == 1
                and set(neighbors) == {atom2, atom4, other_hydrogens[0]}
            )
            if not terminal_pair:
                self._dihedral_symmetries.append(None)
                continue
            if not coord.real_values or not np.isfinite(float(coord.real_values[0])):
                raise ValueError(
                    f"Dihedral {tuple(coord.atom_indices)} has no angle on the input geometry"
                )
            primary = float(coord.real_values[0])
            alternate = copy.deepcopy(coord)
            alternate.atom_indices = [atom1, atom2, atom3, other_hydrogens[0]]
            alternate.compute(self.initial_mol)
            shift = (float(alternate.real_values[0]) - primary + 180.0) % 360.0 - 180.0
            if not np.isfinite(shift) or abs(shift) < 1.0:
                raise ValueError(
                    f"Dihedral {tuple(coord.atom_indices)} hydrogens do not define a symmetry shift"
                )
            self._dihedral_symmetries.append(shift)

    def _replace_search_geometry(self, molecule: Molecule) -> None:
        """Apply later dihedrals to this Cartesian structure."""
        elements = [atom.element for atom in molecule.atoms]
        if elements != [atom.element for atom in self.initial_mol.atoms]:
            raise ValueError("Search geometry does not match the input atom sequence")
        self.initial_mol = molecule
        self.population_generator.molecule = PopulationGenerator._copy_molecule(molecule)
        for coord in self.population_generator.coordinates.elements:
            coord.compute(self.population_generator.molecule)
        self.template_coords = self.population_generator.coordinates
        self._set_dihedral_symmetries()

    def _covalent_radius(self, element: str) -> float:
        if element not in _COVALENT_RADII:
            raise ValueError(f"No covalent radius for element {element}")
        return _COVALENT_RADII[element]

    def _vdw_radius(self, element: str) -> float:
        if element not in _VDW_RADII:
            raise ValueError(f"No van der Waals radius for element {element}")
        return _VDW_RADII[element]

    def _bonds_of(self, molecule: Molecule):
        atoms = molecule.atoms
        bonded = set()
        for i in range(len(atoms)):
            radius_i = self._covalent_radius(atoms[i].element)
            for j in range(i + 1, len(atoms)):
                radius_j = self._covalent_radius(atoms[j].element)
                if atoms[i].distance_to(atoms[j]) < (radius_i + radius_j) * 1.2:
                    bonded.add((i, j))
        return bonded

    def _overlapping_atoms(self, molecule: Molecule):
        """Return a description when two atoms that are not bonded in the seed occupy the same space."""
        if self._bonded_pairs is None:
            raise ValueError("Overlap checks require the seed bond graph")
        atoms = molecule.atoms
        if len(atoms) != len(self.initial_mol.atoms):
            raise ValueError(
                f"Conformer has {len(atoms)} atoms; the seed has {len(self.initial_mol.atoms)}"
            )
        worst = None
        for i in range(len(atoms)):
            radius_i = self._vdw_radius(atoms[i].element)
            for j in range(i + 1, len(atoms)):
                if (i, j) in self._bonded_pairs:
                    continue
                radius_j = self._vdw_radius(atoms[j].element)
                distance = atoms[i].distance_to(atoms[j])
                limit = _OVERLAP_VDW_FRACTION * (radius_i + radius_j)
                if distance < limit and (worst is None or distance < worst[0]):
                    worst = (distance, i, atoms[i].element, j, atoms[j].element, limit)
        if worst is None:
            return None
        distance, i, element_i, j, element_j, limit = worst
        return (
            f"{element_i}{i + 1} and {element_j}{j + 1} are {distance:.3f} A apart, "
            f"inside {_OVERLAP_VDW_FRACTION:g} times the van der Waals sum {limit:.3f} A"
        )

    def _exclude_overlaps(self, population):
        kept = []
        for item in population:
            if not isinstance(item, tuple) or len(item) != 4:
                raise ValueError(
                    "Population conformers must carry coordinates, origin, energy, and history index"
                )
            coords, origin, energy, _history_index = item
            if energy is not None:
                kept.append(item)
                continue
            overlap = self._overlapping_atoms(self.molecule_from_coordinates(coords))
            if overlap is None:
                kept.append(item)
                continue
            self._report(
                f"Excluded {origin} conformer because {overlap}. "
                "Overlapping atoms score positive infinity and leave the population."
            )
        return kept

    def _remember_conformation_energies(self, population) -> None:
        ordered = sorted(population, key=lambda item: (float(item[2]), item[1]))
        rows = []
        for index, item in enumerate(ordered):
            if not isinstance(item, tuple) or len(item) != 4:
                raise ValueError(
                    "Population conformers must carry coordinates, origin, energy, and history index"
                )
            _coords, origin, energy, history_index = item
            if not isinstance(history_index, int) or not 0 <= history_index < len(self._history):
                raise ValueError(f"Configuration {history_index} is not in the GA history")
            parts = [str(history_index)]
            pending = [history_index]
            visited = set()
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                if not isinstance(current, int) or not 0 <= current < len(self._history):
                    raise ValueError(f"Configuration {current} is not in the GA history")
                record = self._history[current]
                if not isinstance(record, dict) or "op" not in record:
                    raise ValueError(f"Configuration {current} has no history operation")
                visited.add(current)
                operation = record["op"]
                if operation == "I":
                    parts.append(f"I,{current}")
                elif operation == "M":
                    if "ancestor" not in record or "dihedral" not in record or "angle" not in record:
                        raise ValueError(
                            f"Mutation {current} is missing ancestor, dihedral, or angle"
                        )
                    ancestor = record["ancestor"]
                    parts.append(
                        f"M,{ancestor},{record['dihedral']},{float(record['angle']):+.6f}"
                    )
                    pending.append(ancestor)
                elif operation == "X":
                    if "a1" not in record or "a2" not in record or "cut" not in record:
                        raise ValueError(f"Crossover {current} is missing ancestors or cut")
                    parts.append(f"X,{record['a1']},{record['a2']},{record['cut']}")
                    pending.append(record["a2"])
                    pending.append(record["a1"])
                else:
                    raise ValueError(f"Unknown GA history operation {operation!r}")
            rows.append({
                "conformation": index + 1,
                "energy": float(energy),
                "origin": origin,
                "history": " < ".join(parts),
            })
        self._conformation_rows = rows

    def _attach_conformation_table(self, chart) -> None:
        if not self._conformation_rows:
            return
        chart.options = {
            "conformations": [dict(row) for row in self._conformation_rows],
        }

    def evaluate_population(self, population: List[InternalCoordinatesList]) -> List[Tuple[float, InternalCoordinatesList]]:
        molecules = [self.molecule_from_coordinates(ind) for ind in population]
        self.evaluated_molecules = self.evaluate_molecules(molecules)
        return [(mol.properties["energy"], ind)
                for mol, ind in zip(self.evaluated_molecules, population)]

    def reproduce(self, best_individuals, target_size):
        return self.population_generator.reproduce(
            best_individuals, target_size, self.prune_rms_thresh, self._history,
        )

    def dihedral_rmsd(self, ind1: InternalCoordinatesList, ind2: InternalCoordinatesList) -> float:
        if self._dihedral_symmetries is None:
            raise ValueError("Dihedral symmetries are unavailable before setup")
        return dihedral_rmsd(ind1, ind2, self._dihedral_symmetries)

    def _report(self, message: str) -> None:
        """Send a message to the task log and the node runner info log."""
        if hasattr(self.node_runner, "log_string"):
            self.node_runner.log(message)
        self.node_runner.info(message)

    def _force_field_discarded(self, molecule) -> bool:
        if molecule.properties.get("force_field_discarded") is not True:
            return False
        gradient = molecule.properties.get("force_field_gradient")
        if gradient is None or not np.isfinite(float(gradient)):
            raise ValueError("Discarded conformer has no force-field gradient")
        self._report(
            "Discarded conformer because the force-field gradient stayed high "
            f"({float(gradient):.6g}) after two repeated optimizations."
        )
        return True

    def _count_origin(self, origin: str, *, elite: bool) -> None:
        counts = self.success_stats.setdefault(origin, [0, 0])
        counts[1] += 1
        if elite:
            counts[0] += 1

    def _record_energy_iteration(self, iteration: int, energies: List[float]) -> None:
        if not energies:
            raise ValueError(f"GA iteration {iteration} produced no energies")
        self.energy_history.append({
            "iteration": int(iteration),
            "min_energy": float(min(energies)),
            "max_energy": float(max(energies)),
        })

    def _ranked_population(self, molecules: List[Molecule], *, limit: Optional[int]) -> MoleculeList:
        if not molecules:
            raise ValueError("Final population is empty")
        for molecule in molecules:
            energy = molecule.properties.get("energy")
            if energy is None or not np.isfinite(energy):
                raise ValueError("Population molecule has no finite energy")
        if limit is not None and limit < 1:
            raise ValueError(f"num_confs limit must be a positive integer, got {limit}")
        ranked = sorted(molecules, key=lambda molecule: float(molecule.properties["energy"]))
        kept = []
        kept_coords = []
        nearest_dropped = None
        examined = 0
        for molecule in ranked:
            examined += 1
            coords = self.coordinates_from_molecule(molecule)
            if kept_coords:
                distance = min(
                    self.dihedral_rmsd(coords, previous) for previous in kept_coords
                )
                if distance < self.prune_rms_thresh:
                    if nearest_dropped is None or distance < nearest_dropped:
                        nearest_dropped = distance
                    continue
            kept.append(molecule)
            kept_coords.append(coords)
            if limit is not None and len(kept) == limit:
                break
        if not kept:
            raise ValueError("Final population is empty")
        dropped = examined - len(kept)
        unexamined = len(ranked) - examined
        if limit is None:
            cap_note = "num_confs cap is not applied"
        else:
            cap_note = (
                f"Stopped at num_confs {limit}; "
                f"{unexamined} higher-energy conformers were not compared."
            )
        self._report(
            f"Conformer prune: {len(ranked)} in, kept {len(kept)}, "
            f"{self._dihedral_prune_phrase(dropped, nearest_dropped)}. "
            f"{cap_note}."
        )
        return _to_molecule_list(kept)

    def _dihedral_prune_phrase(self, dropped: int, nearest) -> str:
        if dropped == 0:
            return (
                f"dropped 0 with dihedral RMSD below prune_rms_thresh {self.prune_rms_thresh}"
            )
        if nearest is None or not np.isfinite(nearest):
            raise ValueError("Dihedral RMSD prune dropped conformers without a distance")
        return (
            f"dropped {dropped} with dihedral RMSD below prune_rms_thresh {self.prune_rms_thresh} "
            f"(nearest {float(nearest):.4f} degrees)"
        )

    def _unique_conformers(self, scored_pop):
        """Keep the lowest-energy conformer of each dihedral neighborhood."""
        kept = []
        nearest_dropped = None
        for energy, coords, origin, history_index in scored_pop:
            if kept:
                distance = min(
                    self.dihedral_rmsd(coords, previous) for _, previous, _, _ in kept
                )
                if distance < self.prune_rms_thresh:
                    if nearest_dropped is None or distance < nearest_dropped:
                        nearest_dropped = distance
                    continue
            kept.append((energy, coords, origin, history_index))
        if not kept:
            raise ValueError("Uniqueness filter removed every conformer")
        dropped = len(scored_pop) - len(kept)
        if dropped and (nearest_dropped is None or not np.isfinite(nearest_dropped)):
            raise ValueError("Uniqueness filter dropped conformers without a dihedral RMSD")
        return kept, {
            "kept": len(kept),
            "dropped": dropped,
            "nearest_dropped_rmsd": nearest_dropped,
        }

    def _select_next_population(
            self,
            scored_parents: List[Tuple[float, InternalCoordinatesList, str]],
            gen: int,
    ):
        """Score new children, keep the lowest-energy member of each dihedral neighborhood, then the lowest pop_size."""
        if not scored_parents:
            raise ValueError("Cannot select from an empty population")
        parents = []
        for energy, coords, origin, history_index in scored_parents:
            if energy is None or not np.isfinite(energy):
                raise ValueError("Parent conformer has no finite energy")
            parents.append((coords, origin, float(energy), history_index))
        proposed = self.reproduce(parents, len(parents) + self.pop_size)
        children = self._exclude_overlaps(proposed[len(parents):])
        child_scored = []
        if children:
            scored = self.evaluate_population([coords for coords, _, _, _ in children])
            if len(scored) != len(children):
                raise ValueError(
                    f"Evaluator returned {len(scored)} energies for {len(children)} children"
                )
            for outcome, (_, origin, _, history_index) in zip(scored, children):
                if outcome is None:
                    continue
                energy, coords = outcome
                if energy is None or not np.isfinite(energy):
                    raise ValueError("Child conformer has no finite energy")
                child_scored.append((float(energy), coords, origin, history_index))
        stats = self.population_generator.reproduction_stats
        if not isinstance(stats, dict):
            raise ValueError("Breeding finished without reproduction statistics")
        for key in ("accepted", "rejected_noop"):
            if key not in stats:
                raise ValueError(f"Reproduction statistics are missing {key}")
        combined = [
            (energy, coords, origin, history_index)
            for coords, origin, energy, history_index in parents
        ] + child_scored
        combined.sort(key=lambda item: item[0])
        unique, prune = self._unique_conformers(combined)
        kept = unique[:self.pop_size]
        nearest_pair = None
        for index, (_, coords, _, _) in enumerate(kept):
            for _, other, _, _ in kept[index + 1:]:
                distance = self.dihedral_rmsd(coords, other)
                if nearest_pair is None or distance < nearest_pair:
                    nearest_pair = distance
                if distance < self.prune_rms_thresh:
                    raise ValueError(
                        f"Generation {gen} kept two conformers {distance:.4f} degrees apart, "
                        f"below prune_rms_thresh {self.prune_rms_thresh}"
                    )
        if nearest_pair is None:
            separation = "one conformer, so no pair to compare"
        else:
            separation = (
                f"closest pair is {float(nearest_pair):.4f} degrees, "
                f"at or above prune_rms_thresh {self.prune_rms_thresh}"
            )
        kept_indexes = {history_index for _, _, _, history_index in kept}
        for _, origin, _, history_index in proposed:
            if history_index in self._counted_history:
                continue
            if not isinstance(history_index, int):
                raise ValueError(f"Configuration {history_index} has no history index")
            self._counted_history.add(history_index)
            self._count_origin(origin, elite=history_index in kept_indexes)
        self._report(
            f"Gen {gen}: parents {len(parents)}; "
            f"children accepted {stats['accepted']}, "
            f"rejected {stats['rejected_noop']} with no dihedral change; "
            f"dihedral-RMSD prune kept {prune['kept']} and "
            f"{self._dihedral_prune_phrase(prune['dropped'], prune['nearest_dropped_rmsd'])}; "
            f"energy sort kept {len(kept)} "
            f"(current population {len(kept)}, maximum pop_size {self.pop_size}); "
            f"{separation}."
        )
        return [
            (coords, origin, energy, history_index)
            for energy, coords, origin, history_index in kept
        ]

    def _publish_energy_chart(self) -> None:
        if not self.energy_history:
            raise ValueError("GA produced no energy history")
        spans = []
        for point in self.energy_history:
            span = float(point["max_energy"]) - float(point["min_energy"])
            if span < 0 or not np.isfinite(span):
                raise ValueError(
                    f"Iteration {point['iteration']} has an invalid energy range {span}"
                )
            spans.append(span)
        range_factor = 10
        min_plotted_iterations = 10
        start = 0
        for index in range(len(spans) - 1):
            previous, current = spans[index], spans[index + 1]
            if previous == 0 and current == 0:
                continue
            if (
                previous == 0
                or current == 0
                or max(previous, current) / min(previous, current) > range_factor
            ):
                start = index + 1
        earliest_start = max(0, len(spans) - min_plotted_iterations)
        if start > earliest_start:
            start = earliest_start
        if start and start != self._energy_plot_start:
            self._report(
                f"Dropped {start} initial iterations from the energy plot; "
                "the min-max energy range changed by more than a factor of 10."
            )
        self._energy_plot_start = start
        plotted_history = self.energy_history[start:]
        data = [
            {
                "iteration": int(point["iteration"]),
                "min-energy": float(point["min_energy"]),
                "max-energy": float(point["max_energy"]),
            }
            for point in plotted_history
        ]
        series = [
            AGLineSeriesConfig(
                type="line",
                xKey="iteration",
                yKey="min-energy",
                title="min-energy",
                data=data,
                stroke="#4ECDC4",
            ),
            AGLineSeriesConfig(
                type="line",
                xKey="iteration",
                yKey="max-energy",
                title="max-energy",
                data=data,
                stroke="#FF6B6B",
            ),
        ]
        chart = getattr(self.node_runner, "energy_chart", None)
        if isinstance(chart, ChartArtifactModel):
            chart.data = data
            chart.series = series
            self._attach_conformation_table(chart)
            return
        self.node_runner.energy_chart = ChartArtifactModel(
            data=data,
            title=AGChartTitleConfig(text="Energy vs iteration"),
            series=series,
            axes=[
                AGChartAxisConfig(type="number", position="bottom", title="iteration"),
                AGChartAxisConfig(type="number", position="left", title="energy"),
            ],
            legend=AGChartLegendConfig(enabled=True, position="right"),
        )
        self._attach_conformation_table(self.node_runner.energy_chart)

    def _publish_energy_histogram(self, energies: List[float]) -> None:
        if not energies:
            raise ValueError("Cannot build an energy histogram for an empty population")
        low, high = min(energies), max(energies)
        bin_count = 20
        if high == low:
            bins = [{"energy": f"{low:.6g}", "count": len(energies)}]
        else:
            width = (high - low) / bin_count
            counts = [0] * bin_count
            for energy in energies:
                index = int((energy - low) / width)
                if index == bin_count:
                    index = bin_count - 1
                counts[index] += 1
            bins = [
                {
                    "energy": f"{low + index * width:.6g} – {low + (index + 1) * width:.6g}",
                    "count": counts[index],
                }
                for index in range(bin_count)
            ]
        series = [
            AGColumnSeriesConfig(
                type="column",
                xKey="energy",
                yKey="count",
                title="count",
                data=bins,
                fill="#45B7D1",
            )
        ]
        chart = getattr(self.node_runner, "energy_histogram", None)
        if isinstance(chart, ChartArtifactModel):
            chart.data = bins
            chart.series = series
            self._attach_conformation_table(chart)
            return
        self.node_runner.energy_histogram = ChartArtifactModel(
            data=bins,
            title=AGChartTitleConfig(text="Energy histogram"),
            series=series,
            axes=[
                AGChartAxisConfig(type="category", position="bottom", title="energy"),
                AGChartAxisConfig(type="number", position="left", title="count"),
            ],
            legend=AGChartLegendConfig(enabled=False),
        )
        self._attach_conformation_table(self.node_runner.energy_histogram)

    def _publish_diversity_chart(self, population: List[InternalCoordinatesList]) -> None:
        if len(population) < 2:
            return
        width = len(population[0].elements)
        if width < 1:
            raise ValueError("Diversity analysis needs dihedral coordinates")
        angles = []
        for individual in population:
            if len(individual.elements) != width:
                raise ValueError("Diversity analysis received inconsistent dihedral lists")
            angles.append([
                coordinate.get_actual_value(coordinate.value)
                for coordinate in individual.elements
            ])
        radians = np.deg2rad(np.asarray(angles, dtype=float))
        differences = radians[:, None, :] - radians[None, :, :]
        differences = (differences + np.pi) % (2 * np.pi) - np.pi
        rmsd = np.sqrt(np.mean(np.square(np.rad2deg(differences)), axis=-1))
        if rmsd.shape != (len(population), len(population)) or not np.isfinite(rmsd).all():
            raise ValueError("Pairwise dihedral RMSD is not finite")
        centered = rmsd - rmsd.mean(axis=0, keepdims=True)
        singular_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        total = float(np.sum(np.square(singular_values)))
        embedding = np.zeros((len(population), 2))
        percents = []
        for rank in range(min(2, singular_vectors.shape[1])):
            embedding[:, rank] = singular_vectors[:, rank] * singular_values[rank]
            percents.append(0.0 if total == 0 else 100.0 * float(singular_values[rank] ** 2) / total)
        while len(percents) < 2:
            percents.append(0.0)
        points = [
            {"pc1": float(embedding[index, 0]), "pc2": float(embedding[index, 1])}
            for index in range(len(population))
        ]
        series = [
            AGScatterSeriesConfig(
                type="scatter",
                xKey="pc1",
                yKey="pc2",
                title="conformers",
                data=points,
                fill="#2A9D8F",
            )
        ]
        title = AGChartTitleConfig(
            text=f"Dihedral RMSD PCA (PC1 {percents[0]:.1f}%, PC2 {percents[1]:.1f}%)"
        )
        chart = getattr(self.node_runner, "diversity_chart", None)
        if isinstance(chart, ChartArtifactModel):
            chart.title = title
            chart.data = points
            chart.series = series
            self._attach_conformation_table(chart)
            return
        self.node_runner.diversity_chart = ChartArtifactModel(
            data=points,
            title=title,
            series=series,
            axes=[
                AGChartAxisConfig(type="number", position="bottom", title="PC1"),
                AGChartAxisConfig(type="number", position="left", title="PC2"),
            ],
            legend=AGChartLegendConfig(enabled=False),
        )
        self._attach_conformation_table(self.node_runner.diversity_chart)

    def _publish_dihedral_histograms(self, population: List[InternalCoordinatesList]) -> None:
        if not population:
            raise ValueError("Cannot build dihedral histograms for an empty population")
        width = len(population[0].elements)
        if width == 0:
            return
        if self._dihedral_histogram_names and len(self._dihedral_histogram_names) != width:
            raise ValueError(
                f"Dihedral histogram count changed from {len(self._dihedral_histogram_names)} to {width}"
            )
        samples = [[] for _ in range(width)]
        labels = []
        for individual in population:
            if len(individual.elements) != width:
                raise ValueError("Dihedral histograms received inconsistent dihedral lists")
            for index, coordinate in enumerate(individual.elements):
                angle = float(coordinate.get_actual_value(coordinate.value))
                if not np.isfinite(angle):
                    raise ValueError(f"Dihedral {index} has a non-finite angle {angle}")
                samples[index].append(angle)
                if len(labels) == index:
                    labels.append("-".join(str(atom) for atom in coordinate.atom_indices))
        bin_count = 12
        bin_width = 30.0
        fills = ("#45B7D1", "#F4A261", "#2A9D8F", "#E76F51", "#6D597A", "#457B9D")
        for index, angles in enumerate(samples):
            counts = [0] * bin_count
            for angle in angles:
                wrapped = (angle + 180.0) % 360.0 - 180.0
                bin_index = int((wrapped + 180.0) / bin_width)
                if bin_index == bin_count:
                    bin_index = 0
                counts[bin_index] += 1
            bins = [
                {
                    "angle": f"{-180.0 + bin_index * bin_width:.6g} – "
                             f"{-180.0 + (bin_index + 1) * bin_width:.6g}",
                    "count": counts[bin_index],
                }
                for bin_index in range(bin_count)
            ]
            series = [
                AGColumnSeriesConfig(
                    type="column",
                    xKey="angle",
                    yKey="count",
                    title="count",
                    data=bins,
                    fill=fills[index % len(fills)],
                )
            ]
            name = f"dihedral_histogram_{index}"
            title = AGChartTitleConfig(text=f"Dihedral {labels[index]}")
            chart = getattr(self.node_runner, name, None)
            if isinstance(chart, ChartArtifactModel):
                chart.title = title
                chart.data = bins
                chart.series = series
            else:
                chart = ChartArtifactModel(
                    data=bins,
                    title=title,
                    series=series,
                    axes=[
                        AGChartAxisConfig(type="category", position="bottom", title="angle"),
                        AGChartAxisConfig(type="number", position="left", title="count"),
                    ],
                    legend=AGChartLegendConfig(enabled=False),
                )
                setattr(self.node_runner, name, chart)
            self._attach_conformation_table(chart)
            if name not in self._dihedral_histogram_names:
                self._dihedral_histogram_names.append(name)

    def _persist_live_artifacts(self) -> None:
        from odmantic import ObjectId
        from simstack.core.context import context

        if not context.initialized:
            return
        task_id = getattr(self.node_runner, "task_id", None)
        if task_id is None:
            return
        if self.artifact_loop is None:
            raise RuntimeError("Cannot publish GA artifacts while running without the node event loop")
        if not self.artifact_loop.is_running():
            raise RuntimeError("The node event loop is not running, so GA artifacts cannot be published")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_on_loop = False
        else:
            running_on_loop = True
        if running_on_loop:
            raise RuntimeError("Publish GA artifacts from the worker thread, not the event loop")
        parent_id = task_id if isinstance(task_id, ObjectId) else ObjectId(str(task_id))
        charts = []
        for name in ("energy_chart", "energy_histogram", "diversity_chart", *self._dihedral_histogram_names):
            chart = getattr(self.node_runner, name, None)
            if chart is None:
                continue
            if not isinstance(chart, ChartArtifactModel):
                raise ValueError(f"GA artifact {name} is {type(chart).__name__}, expected ChartArtifactModel")
            chart.parent_id = parent_id
            charts.append(chart)

        async def save_charts():
            for chart in charts:
                await context.db.save(chart)

        asyncio.run_coroutine_threadsafe(save_charts(), self.artifact_loop).result()

    def _publish_run_report(self, molecules: List[Molecule]) -> None:
        self._publish_energy_chart()
        if not self._conformation_rows:
            raise ValueError("GA produced no conformation energies")
        self._publish_energy_histogram([
            float(row["energy"]) for row in self._conformation_rows
        ])
        if self.population_generator is None:
            raise ValueError("Cannot measure dihedral histograms before the population generator exists")
        self._publish_dihedral_histograms([
            self.coordinates_from_molecule(molecule) for molecule in molecules
        ])
        table = SimpleTable(name="Population change success rates")
        table.add_column("change", SimpleTableColumnType.STRING)
        table.add_column("produced", SimpleTableColumnType.NUMBER)
        table.add_column("elite_survivors", SimpleTableColumnType.NUMBER)
        table.add_column("success_rate", SimpleTableColumnType.NUMBER)
        self._report("\n--- GA Success Rates (changes kept after selection) ---")
        for origin, counts in self.success_stats.items():
            if len(counts) != 2:
                raise ValueError(f"Success counts for {origin} are not usable: {counts}")
            survived, produced = counts
            if produced == 0 and survived == 0:
                continue
            if produced <= 0 or survived < 0 or survived > produced:
                raise ValueError(f"Success counts for {origin} are not usable: {counts}")
            rate = survived / produced
            self._report(f"{origin:<15}: {rate * 100:6.2f}% ({survived}/{produced})")
            table.add_row({
                "change": origin,
                "produced": produced,
                "elite_survivors": survived,
                "success_rate": rate,
            })
        self.node_runner.operator_stats = table
        energies = SimpleTable(name="Conformation energies")
        energies.add_column("conformation", SimpleTableColumnType.NUMBER)
        energies.add_column("origin", SimpleTableColumnType.STRING)
        energies.add_column("energy", SimpleTableColumnType.NUMBER)
        energies.add_column("history", SimpleTableColumnType.STRING)
        for row in self._conformation_rows:
            if "history" not in row:
                raise ValueError(f"Conformation {row.get('conformation')} has no history")
            energies.add_row({
                "conformation": row["conformation"],
                "origin": row["origin"],
                "energy": row["energy"],
                "history": row["history"],
            })
        self.node_runner.conformation_energies = energies
        self._persist_live_artifacts()
        self._report(f"Returning {len(molecules)} population molecules.")

    def run(self) -> MoleculeList:
        self.setup()
        loaded = self.load_state() if self.restart else None
        if loaded is None:
            minimized = self.evaluate_molecules([self.initial_mol], optimize=True)
            if len(minimized) != 1:
                raise ValueError(
                    f"Minimization of the input geometry returned {len(minimized)} molecules"
                )
            molecule = minimized[0]
            if self._force_field_discarded(molecule):
                raise ValueError(
                    "Minimization of the input geometry left a high force-field gradient"
                )
            self._replace_search_geometry(molecule)
            self._report(
                "Minimized the input geometry with "
                f"{type(self.evaluator).__name__} to "
                f"{float(molecule.properties['energy']):.6g} kcal/mol before the search."
            )
        self._report("\n--- GA Parameters ---")
        self._report(f"Atoms:          {len(self.initial_mol.atoms)}")
        self._report(f"Population:     {self.pop_size}")
        self._report("Seed:           minimized input geometry")
        self._report(f"Generations:    {self.generations}")
        self._report(f"Mutation Rate:  {self.mutation_rate}")
        self._report(f"Crossover Rate: {1 - self.mutation_rate}")
        self._report(f"Dihedral Step:  {self.dihedral_interval}")
        self._report(f"Evaluator:      {type(self.evaluator).__name__} (Max Iters: {self.max_iters})")
        self._report(f"Parallel children: {self.parallel_children}")
        self._report(f"Match double bonds: {self.match_double_bonds}")
        self._report(
            f"Rotatable bond range: {self.rotatable_bond_min} .. {self.rotatable_bond_max}"
        )
        self._report(f"Seed:           {self.seed}")
        self._report(f"Smart Opt:      {self.smart_opt}")
        self._report(f"Restart:        {self.restart}")
        if hasattr(self, "n_prune"):
            self._report(f"Prune Every:    {getattr(self, 'n_prune')}")
        if hasattr(self, "prune_rms_thresh"):
            self._report(f"Prune RMS:      {getattr(self, 'prune_rms_thresh')}")
        self._report(
            "Population evolution: the input geometry is minimized with the selected "
            "method before the first generation. pop_size "
            f"({self.pop_size}) is the maximum live population. Each generation proposes "
            "that many children, scores them, and keeps the lower-energy conformer when "
            "a child is within prune_rms_thresh "
            f"({self.prune_rms_thresh} degrees dihedral RMSD) of a parent or another child. "
            "Parents and surviving children are combined, sorted by energy, and cut at "
            "pop_size. The live population can grow back up to pop_size; the maximum does "
            "not shrink. The standard GA returns every conformer that survived the last "
            "selection, with the single-point energy used during the search, and does not "
            "relax those geometries. num_confs caps the returned list of ga-min and "
            "ga-select. ga-min compares optimized geometries, so starts that relax into "
            "one well collapse there. ga-select also drops conformers above "
            "min energy + 0.1*|min energy| before breeding. "
            "When a force-field minimization stops with a gradient above its force "
            "tolerance, it is repeated twice, and the conformer is discarded if the "
            "gradient is still high."
        )
        self._report(f"Rotatable Bonds: {len(self.template_coords.elements)}")
        self._report(f"Dihedrals:       {self.dihedrals}")
        self._report(f"Dihedral Types:  {self.dihedral_types}")
        self._report("---------------------\n")

        if not self.template_coords.elements:
            if loaded is not None:
                raise ValueError("Restarted GA has no rotatable bonds")
            self._report("No rotatable bonds found.")
            evaluated = self.evaluate_molecules(
                [self.initial_mol], optimize=not isinstance(self, StandardGA),
            )
            evaluated = [
                molecule for molecule in evaluated if not self._force_field_discarded(molecule)
            ]
            if not evaluated:
                raise ValueError(
                    "Force-field optimization discarded the conformer because its gradient "
                    "stayed high after two repeated optimizations"
                )
            ordered = sorted(evaluated, key=lambda molecule: float(molecule.properties["energy"]))
            if len(ordered) != 1:
                raise ValueError(
                    f"A molecule with no rotatable bonds produced {len(ordered)} conformers"
                )
            if self._history:
                raise ValueError("GA history was already started")
            self._history.append({"op": "I", "index": 0})
            self._conformation_rows = [
                {
                    "conformation": 1,
                    "energy": float(ordered[0].properties["energy"]),
                    "origin": "initial",
                    "history": "0 < I,0",
                }
            ]
            self._record_energy_iteration(0, [mol.properties["energy"] for mol in evaluated])
            molecules = self._ranked_population(evaluated, limit=self.num_confs)
            self._publish_run_report(list(molecules))
            return molecules

        start_gen = 0
        population = None
        if loaded is not None:
            population, start_gen = loaded
            start_gen += 1

        if population is None:
            generated = self.population_generator.generate(1)
            if len(generated) != 1:
                raise ValueError(f"Seed generation returned {len(generated)} conformers")
            coords, origin, energy = generated[0]
            if self._history:
                raise ValueError("GA history was already started")
            self._history.append({"op": "I", "index": 0})
            population = [(coords, origin, energy, 0)]
            self._report(
                "Seed population: the minimized input geometry alone. "
                "Later conformers come from mutation and crossover."
            )

        t1 = time.perf_counter()
        for gen in range(start_gen, self.generations + 1):
            population = self._exclude_overlaps(population)
            if not population:
                raise ValueError(f"Generation {gen} has no conformers after excluding overlaps")
            pending = [index for index, (_, _, energy, _) in enumerate(population) if energy is None]
            if pending:
                scored_pop = self.evaluate_population([population[index][0] for index in pending])
                if len(scored_pop) != len(pending):
                    raise ValueError(
                        f"Evaluator returned {len(scored_pop)} energies for {len(pending)} individuals"
                    )
                discarded = []
                for index, outcome in zip(pending, scored_pop):
                    if outcome is None:
                        discarded.append(index)
                        continue
                    energy, coords = outcome
                    if energy is None or not np.isfinite(energy):
                        raise ValueError("Population conformer has no finite energy")
                    population[index] = (
                        coords, population[index][1], float(energy), population[index][3],
                    )
                if discarded:
                    discarded_indexes = set(discarded)
                    population = [
                        item for index, item in enumerate(population) if index not in discarded_indexes
                    ]
                if not population:
                    raise ValueError(
                        "Generation "
                    f"{gen} has no conformers after discarding overlaps or high force-field gradients"
                    )
            for _, _, energy, _ in population:
                if energy is None or not np.isfinite(energy):
                    raise ValueError("Population conformer has no finite energy")
            scored_with_labels = [
                (energy, coords, origin, history_index)
                for coords, origin, energy, history_index in population
            ]
            population = self.selection(scored_with_labels, gen)
            energies = []
            scored_coords = []
            for coords, _, energy, _ in population:
                if energy is None or not np.isfinite(energy):
                    raise ValueError("Population conformer has no finite energy")
                energies.append(energy)
                scored_coords.append(coords)
            self._remember_conformation_energies(population)
            self._record_energy_iteration(gen, energies)
            self._publish_energy_chart()
            self._publish_energy_histogram(energies)
            self._publish_diversity_chart(scored_coords)
            self._publish_dihedral_histograms(scored_coords)
            self._persist_live_artifacts()
            self.post_generation_hook([coords for coords, _, _, _ in population], gen)

            # Save state at the end of each generation (or every n generations)
            if gen != 0 and gen % 5 == 0 or gen == self.generations:
                self.save_state(population, gen)

        self.timing["GA Loop"] = time.perf_counter() - t1
        molecules = self.finalize(population)
        returned = list(molecules)
        cap = self.pop_size if isinstance(self, StandardGA) else self.num_confs
        if not 1 <= len(returned) <= cap:
            raise ValueError(
                f"Expected between 1 and {cap} population molecules, got {len(returned)}"
            )
        self._publish_run_report(returned)
        return molecules

    def selection(
            self,
            scored_pop: List[Tuple[float, InternalCoordinatesList, str]],
            gen: int
    ):
        scored_pop.sort(key=lambda x: x[0])
        return self._select_next_population(scored_pop, gen)

    def post_generation_hook(self, population: List[InternalCoordinatesList], gen: int):
        pass

    def finalize(self, population) -> MoleculeList:
        t2 = time.perf_counter()
        self.evaluate_population([coords for coords, _, _, _ in population])
        final_ranked = self._ranked_population(self.evaluated_molecules, limit=self.num_confs)

        self.timing["Finalization"] = time.perf_counter() - t2

        self.smart_optimizer.report()

        output_prefix = f"{self.__class__.__name__.lower()}_diversity"
        plot_conformer_diversity(final_ranked, output_prefix=output_prefix)

        if self.profile:
            self._report(f"\n--- {self.__class__.__name__} CPU Profile ---")
            for k, v in self.timing.items(): self._report(f"{k:<25}: {v:.4f}s")
        return final_ranked


class StandardGA(BaseGA):
    """Minimize the input, then score generated geometries without relaxing them."""

    def finalize(self, population) -> MoleculeList:
        ordered = sorted(population, key=lambda item: (float(item[2]), item[1]))
        molecules = []
        for coords, _, energy, _history_index in ordered:
            if energy is None or not np.isfinite(float(energy)):
                raise ValueError("Population molecule has no finite energy")
            molecule = self.molecule_from_coordinates(coords)
            molecule.properties["energy"] = float(energy)
            molecule.properties["energy_unit"] = "kcal/mol"
            molecules.append(molecule)
        self.best_energy_seen = molecules[0].properties["energy"]
        self._report(
            f"Final population: {len(molecules)} conformers scored before UFF relaxation, "
            f"returning {len(molecules)}. UFF relaxation is not applied."
        )
        return _to_molecule_list(molecules)


class MinimizingGA(BaseGA):
    """Relax every generation through the selected molecule evaluator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relaxed = {}

    def evaluate_population(self, population: List[InternalCoordinatesList]) -> List[Tuple[float, InternalCoordinatesList]]:
        molecules = [self.molecule_from_coordinates(ind) for ind in population]
        if self.smart_opt:
            starting = self.evaluate_molecules(molecules)
            partial = self.evaluate_molecules(molecules, optimize=True,
                                              max_iters=max(1, self.max_iters // 10))
            keep = []
            for index, (start, current) in enumerate(zip(starting, partial)):
                if current.properties.get("force_field_discarded") is True:
                    continue
                self.smart_optimizer.stats["total"] += 1
                delta = start.properties["energy"] - current.properties["energy"]
                if not self.smart_optimizer.should_discard(
                    delta, current.properties["energy"], self.best_energy_seen
                ):
                    keep.append(index)
            completed = self.evaluate_molecules([partial[i] for i in keep], optimize=True)
            optimized = list(partial)
            for index, result in zip(keep, completed):
                start = starting[index].properties["energy"]
                intermediate = partial[index].properties["energy"]
                self.smart_optimizer.collect(start - intermediate, intermediate,
                                             start - result.properties["energy"])
                optimized[index] = result
        else:
            optimized = self.evaluate_molecules(molecules, optimize=True)
        self.evaluated_molecules = [
            molecule for molecule in optimized if not self._force_field_discarded(molecule)
        ]
        if self.evaluated_molecules:
            self.best_energy_seen = min(
                self.best_energy_seen,
                min(float(molecule.properties["energy"]) for molecule in self.evaluated_molecules),
            )
        paired = []
        kept = {id(molecule) for molecule in self.evaluated_molecules}
        for molecule in optimized:
            if id(molecule) not in kept:
                paired.append(None)
                continue
            overlap = self._overlapping_atoms(molecule)
            if overlap is not None:
                self._report(
                    f"Excluded optimized conformer because {overlap}. "
                    "Overlapping atoms score positive infinity and leave the population."
                )
                paired.append(None)
                continue
            coords = self.coordinates_from_molecule(molecule)
            self.relaxed[id(coords)] = molecule
            paired.append((float(molecule.properties["energy"]), coords))
        return paired

    def finalize(self, population) -> MoleculeList:
        pending = [index for index, (_, _, energy, _) in enumerate(population) if energy is None]
        if pending:
            scored = self.evaluate_population([population[index][0] for index in pending])
            if len(scored) != len(pending):
                raise ValueError(
                    f"Evaluator returned {len(scored)} energies for {len(pending)} individuals"
                )
            discarded = []
            for index, outcome in zip(pending, scored):
                if outcome is None:
                    discarded.append(index)
                    continue
                energy, coords = outcome
                population[index] = (
                    coords, population[index][1], float(energy), population[index][3],
                )
            if discarded:
                discarded_indexes = set(discarded)
                population = [
                    item for index, item in enumerate(population) if index not in discarded_indexes
                ]
            if not population:
                raise ValueError(
                    "Final population has no conformers after discarding overlaps "
                    "or high force-field gradients"
                )
        molecules = []
        for coords, _, _, _ in population:
            molecule = self.relaxed.get(id(coords))
            if molecule is None:
                raise ValueError("Minimizing GA is missing the relaxed geometry for a carried conformer")
            molecules.append(molecule)
        return self._ranked_population(molecules, limit=self.num_confs)


class DiversityGA(BaseGA):
    def __init__(self, *args, n_prune: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_prune = n_prune
        self.rmsd_history = []
        if n_prune < 1:
            raise ValueError("n_prune must be positive")

    def selection(
            self,
            scored_pop: List[Tuple[float, InternalCoordinatesList, str]],
            gen: int
    ):
        if not scored_pop:
            raise RuntimeError("DiversityGA.selection received an empty population.")

        scored_pop.sort(key=lambda x: x[0])
        min_energy = scored_pop[0][0]
        threshold = min_energy + 0.1 * abs(min_energy)
        accepted = [item for item in scored_pop if item[0] <= threshold]
        if not accepted:
            accepted = [scored_pop[0]]
        energy_dropped = len(scored_pop) - len(accepted)
        self._report(
            f"Gen {gen}: energy window (min + 0.1*|min| = {threshold:.6g}) "
            f"dropped {energy_dropped} of {len(scored_pop)}."
        )
        new_population = self._select_next_population(accepted, gen)

        before_scores = self.get_pairwise_rmsd_scores([coords for _, coords, _, _ in accepted])
        after_scores = self.get_pairwise_rmsd_scores([coords for coords, _, _, _ in new_population])
        self.rmsd_history.append({
            "gen": gen,
            "high_before": float(np.max(before_scores)) if len(before_scores) > 0 else 0.0,
            "low_before": float(np.min(before_scores)) if len(before_scores) > 0 else 0.0,
            "high_after": float(np.max(after_scores)) if len(after_scores) > 0 else 0.0,
            "low_after": float(np.min(after_scores)) if len(after_scores) > 0 else 0.0,
        })
        self._report(
            f"Gen {gen:3d} | "
            f"RMSD before [low={np.min(before_scores) if len(before_scores) > 0 else 0:8.4f}, "
            f"high={np.max(before_scores) if len(before_scores) > 0 else 0:8.4f}] | "
            f"after [low={np.min(after_scores) if len(after_scores) > 0 else 0:8.4f}, "
            f"high={np.max(after_scores) if len(after_scores) > 0 else 0:8.4f}] | "
            f"Min energy = {min_energy:10.4f}, Acc = {len(accepted):3d}"
        )
        return new_population

    def get_pairwise_rmsd_scores(self, pop: Union[np.ndarray, List[InternalCoordinatesList]]) -> np.ndarray:
        """
        Vectorized pairwise RMSD calculation using NumPy broadcasting.
        """
        if isinstance(pop, list):
            if not pop:
                return np.zeros(0)
            pop = np.array([[c.get_actual_value(c.value) for c in ind.elements] for ind in pop])

        if len(pop) <= 1:
            return np.zeros(len(pop))

        # Convert to radians for circularity handling
        data = np.deg2rad(pop)  # (N, D)

        # Pairwise differences with broadcasting: (N, 1, D) - (1, N, D) -> (N, N, D)
        diffs = data[:, np.newaxis, :] - data[np.newaxis, :, :]

        # Periodic wrap: (diff + pi) % (2*pi) - pi
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

        # RMSD calculation: sqrt(mean(square(diffs_deg)))
        diffs_deg = np.rad2deg(diffs)
        dist_matrix = np.sqrt(np.mean(np.square(diffs_deg), axis=-1))

        # Return average distance (ignoring self-distance)
        return np.sum(dist_matrix, axis=1) / (len(pop) - 1)

    def post_generation_hook(self, population: List[InternalCoordinatesList], gen: int, force_prune: bool = False):
        if gen > 0 and gen % 50 == 0:
            self.plot_rmsd_evolution(gen)
            self.plot_dihedral_pca(population, gen)

        if (gen > 0 and gen % self.n_prune == 0) or force_prune or not self.best_stored:
            molecules = [self.molecule_from_coordinates(ind) for ind in population]
            temp_ranked = self.evaluate_molecules(molecules, optimize=True)
            if temp_ranked:
                self.best_energy_seen = min(self.best_energy_seen,
                                           min(m.properties["energy"] for m in temp_ranked))

            combined = list(self.best_stored) + temp_ranked
            self.best_stored = self._ranked_population(combined, limit=None)
            self.write_conformers_xyz(self.best_stored, gen)

    def plot_rmsd_evolution(self, gen: int):
        import matplotlib.pyplot as plt
        import pandas as pd
        h = self.rmsd_history

        # Save RMSD evolution to CSV
        rmsd_evol_df = pd.DataFrame(h)
        rmsd_evol_df.to_csv("rmsd_evolution.csv", index=False)
        self._report(" RMSD")

        gens = [x["gen"] for x in h]
        plt.figure(figsize=(10, 6))
        plt.fill_between(gens, [x["low_before"] for x in h], [x["high_before"] for x in h], color="lightsteelblue",
                         alpha=0.4, label="Range (before)")
        plt.fill_between(gens, [x["low_after"] for x in h], [x["high_after"] for x in h], color="navajowhite",
                         alpha=0.6, label="Range (after)")
        plt.plot(gens, [x["high_before"] for x in h], color="steelblue", linestyle="--", linewidth=0.8)
        plt.plot(gens, [x["low_before"] for x in h], color="steelblue", linestyle="--", linewidth=0.8)
        plt.plot(gens, [x["high_after"] for x in h], color="darkorange", linestyle="-", linewidth=1.0)
        plt.plot(gens, [x["low_after"] for x in h], color="darkorange", linestyle="-", linewidth=1.0)
        plt.xlabel("Generation")
        plt.ylabel("Dihedral RMSD")
        plt.title(f"RMSD Evolution (Gen {gen})")
        plt.legend()
        plt.tight_layout()
        plt.savefig("rmsd_evolution.png", dpi=150)
        plt.close()

    def plot_dihedral_pca(self, pop: List[InternalCoordinatesList], gen: int):
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.decomposition import PCA
        if not pop: return
        data = np.array([[c.get_actual_value(c.value) for c in ind.elements] for ind in pop])
        rad = np.deg2rad(data)
        features = np.column_stack([np.sin(rad), np.cos(rad)])
        n = min(2, features.shape[1])
        pca = PCA(n_components=n)
        res = pca.fit_transform(features)

        # Save dihedral PCA to CSV
        out = Path("ga_rmsd_debug")
        out.mkdir(parents=True, exist_ok=True)
        pca_df = pd.DataFrame(res, columns=[f"PC{i + 1}" for i in range(res.shape[1])])
        pca_csv = out / f"dihedral_pca_gen_{gen:04d}.csv"
        pca_df.to_csv(pca_csv, index=False)
        self._report(f" PCA: {pca_csv}")

        plt.figure(figsize=(8, 6))
        if n == 2:
            plt.scatter(res[:, 0], res[:, 1], alpha=0.6, edgecolors="w", color="mediumseagreen")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        else:
            plt.scatter(res[:, 0], np.zeros_like(res[:, 0]), alpha=0.6, edgecolors="w", color="mediumseagreen")
        plt.title(f"Dihedral PCA - Generation {gen}")
        plt.tight_layout()
        plt.savefig(out / f"dihedral_pca_gen_{gen:04d}.png", dpi=150)
        plt.close()

    def write_conformers_xyz(self, conformers: MoleculeList, gen: int):
        with open(f"conformers-{gen:04d}.xyz", "w") as f:
            for i, mol in enumerate(conformers):
                energy = mol.properties.get("energy", 0.0)
                f.write(f"{len(mol.atoms)}\nConformer {i} - Energy: {energy:.4f} kcal/mol\n")
                for atom in mol.atoms:
                    f.write(f"{atom.element} {atom.x:10.5f} {atom.y:10.5f} {atom.z:10.5f}\n")

    def finalize(self, population) -> MoleculeList:
        if not self.best_stored or self.generations % self.n_prune != 0:
            self.post_generation_hook(
                [coords for coords, _, _, _ in population], self.generations, force_prune=True,
            )

        output_prefix = f"{self.__class__.__name__.lower()}"
        plot_conformer_diversity(self.best_stored, output_prefix=output_prefix)

        self.smart_optimizer.report()

        if self.profile:
            self._report(f"\n--- {self.__class__.__name__} CPU Profile ---")
            for k, v in self.timing.items(): self._report(f"{k:<25}: {v:.4f}s")

        self._report("")
        return self._ranked_population(list(self.best_stored), limit=self.num_confs)


def generate_ga_conformers(initial_mol: Molecule, **kwargs):
    return StandardGA(initial_mol, **kwargs).run()


def generate_ga_min_conformers(initial_mol: Molecule, **kwargs):
    return MinimizingGA(initial_mol, **kwargs).run()


def generate_ga_select_conformers(initial_mol: Molecule, **kwargs):
    return DiversityGA(initial_mol, **kwargs).run()
