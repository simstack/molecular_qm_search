from __future__ import annotations
import numpy as np
import time
import pickle
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

from molecular_qm_models import Molecule, MoleculeList, InternalCoordinatesList, prune_conformers
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType
from simstack.core.node_runner import NodeRunner

from .ga_population import PopulationGenerator
from molecular_qm_search.optimization.lib.ga_evaluation import MoleculeEvaluator, make_evaluator, validate_results
from molecular_qm_search.optimization.lib.ga_smart_optimizer import SmartOptimizer
from molecular_qm_search.optimization.models.ga_models import GAOptimizationMethod


def plot_conformer_diversity(conformers: MoleculeList, output_prefix: str = "") -> None:
    """Helper to plot conformer diversity if needed."""
    pass


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
            crossover_rate: float = 0.5,
            dihedral_interval: float = 30.0,
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
    ):
        self.num_confs = num_confs
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dihedral_interval = dihedral_interval
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
        self.success_stats = {"mutation": [0, 0], "crossover": [0, 0], "copy": [0, 0], "initial": [0, 0],
                              "ga-select": [0, 0]}

        if pop_size < 1 or num_confs < 1 or generations < 0:
            raise ValueError("Population and conformer counts must be positive; generations cannot be negative")
        if not 0 <= mutation_rate <= 1 or not 0 <= crossover_rate <= 1:
            raise ValueError("Mutation and crossover rates must lie in [0, 1]")
        if max_iters < 0 or parallel_children < 0 or dihedral_interval < 0:
            raise ValueError("Iteration, parallel children, and dihedral-step values cannot be negative")
        if rotatable_bond_max <= rotatable_bond_min:
            raise ValueError(
                f"rotatable_bond_max ({rotatable_bond_max}) must be greater than "
                f"rotatable_bond_min ({rotatable_bond_min})"
            )
        self.evaluator = evaluator if evaluator is not None else make_evaluator(
            optimization_method, parallel_children=parallel_children,
            backend_options=backend_options,
        )
        self.input_coordinates = coordinates
        self.population_generator = None
        self.evaluated_molecules: List[Molecule] = []

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

    def save_state(self, population: List[Tuple[InternalCoordinatesList, str]], gen: int):
        state = {
            "population": population,
            "gen": gen,
            "best_stored": self.best_stored,
            "success_stats": self.success_stats,
            "best_energy_seen": self.best_energy_seen,
            "random_state": self.population_generator.random.getstate()
        }
        # DiversityGA specific data
        if hasattr(self, "rmsd_history"):
            state["rmsd_history"] = self.rmsd_history

        with open("ga_state.pkl", "wb") as f:
            pickle.dump(state, f)
        self.node_runner.info(f"Saved GA state of generation {gen}.")

    def load_state(self) -> Optional[Tuple[List[Tuple[InternalCoordinatesList, str]], int]]:
        if not Path("ga_state.pkl").exists():
            return None
        try:
            with open("ga_state.pkl", "rb") as f:
                state = pickle.load(f)
            if "random_state" in state:
                self.population_generator.random.setstate(state["random_state"])
            self.best_stored = state.get("best_stored", MoleculeList())
            self.success_stats = state.get("success_stats", self.success_stats)
            self.best_energy_seen = state.get("best_energy_seen", float('inf'))
            if hasattr(self, "rmsd_history") and "rmsd_history" in state:
                self.rmsd_history = state["rmsd_history"]
            self.node_runner.info(f"Loaded GA state from generation {state['gen']}.")
            return state["population"], state["gen"]
        except Exception as e:
            self.node_runner.error(f"Failed to load GA state: {e}")
            return None

    def setup(self):
        """Prepare internal coordinates without scoring or optimizing geometry."""
        t0 = time.perf_counter()
        if self.initial_mol is None:
            raise ValueError("Provide an initial Molecule")
        self.population_generator = PopulationGenerator(
            self.initial_mol, coordinates=self.input_coordinates, seed=self.seed,
            mutation_rate=self.mutation_rate, crossover_rate=self.crossover_rate,
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
        self.timing["Setup"] = time.perf_counter() - t0

    def evaluate_population(self, population: List[InternalCoordinatesList]) -> List[Tuple[float, InternalCoordinatesList]]:
        molecules = [self.molecule_from_coordinates(ind) for ind in population]
        self.evaluated_molecules = self.evaluate_molecules(molecules)
        return [(mol.properties["energy"], ind)
                for mol, ind in zip(self.evaluated_molecules, population)]

    def reproduce(self, best_individuals, target_size):
        return self.population_generator.reproduce(best_individuals, target_size)

    def dihedral_rmsd(self, ind1: Union[InternalCoordinatesList, np.ndarray], ind2: Union[InternalCoordinatesList, np.ndarray]) -> float:
        """
        Vectorized dihedral RMSD calculation using NumPy.
        """
        if isinstance(ind1, InternalCoordinatesList):
            arr1 = np.array([c.get_actual_value(c.value) for c in ind1.elements])
        else:
            arr1 = np.asarray(ind1)
        if isinstance(ind2, InternalCoordinatesList):
            arr2 = np.array([c.get_actual_value(c.value) for c in ind2.elements])
        else:
            arr2 = np.asarray(ind2)
        diffs = np.abs(arr1 - arr2) % 360
        diffs = np.where(diffs > 180, 360 - diffs, diffs)
        return float(np.sqrt(np.mean(np.square(diffs))))

    def run(self) -> MoleculeList:
        self.setup()
        self.node_runner.info("\n--- GA Parameters ---")
        self.node_runner.info(f"Atoms:          {len(self.initial_mol.atoms)}")
        self.node_runner.info(f"Population:     {self.pop_size}")
        self.node_runner.info(f"Generations:    {self.generations}")
        self.node_runner.info(f"Mutation Rate:  {self.mutation_rate}")
        self.node_runner.info(f"Crossover Rate: {self.crossover_rate}")
        self.node_runner.info(f"Dihedral Step:  {self.dihedral_interval}")
        self.node_runner.info(f"Evaluator:      {type(self.evaluator).__name__} (Max Iters: {self.max_iters})")
        self.node_runner.info(f"Parallel children: {self.parallel_children}")
        self.node_runner.info(f"Match double bonds: {self.match_double_bonds}")
        self.node_runner.info(
            f"Rotatable bond range: {self.rotatable_bond_min} .. {self.rotatable_bond_max}"
        )
        self.node_runner.info(f"Seed:           {self.seed}")
        self.node_runner.info(f"Smart Opt:      {self.smart_opt}")
        self.node_runner.info(f"Restart:        {self.restart}")
        if hasattr(self, "n_prune"):
            self.node_runner.info(f"Prune Every:    {getattr(self, 'n_prune')}")
        if hasattr(self, "prune_rms_thresh"):
            self.node_runner.info(f"Prune RMS:      {getattr(self, 'prune_rms_thresh')}")
        self.node_runner.info(f"Rotatable Bonds: {len(self.template_coords.elements)}")
        self.node_runner.info(f"Dihedrals:       {self.dihedrals}")
        self.node_runner.info(f"Dihedral Types:  {self.dihedral_types}")
        self.node_runner.info("---------------------\n")

        if not self.template_coords.elements:
            self.node_runner.info("No rotatable bonds found.")
            return _to_molecule_list(self.evaluate_molecules([self.initial_mol], optimize=True))

        start_gen = 0
        population = None
        if self.restart:
            loaded = self.load_state()
            if loaded:
                population, start_gen = loaded
                start_gen += 1  # Start from the next generation

        if population is None:
            population = self.population_generator.generate(self.pop_size)

        t1 = time.perf_counter()
        for gen in range(start_gen, self.generations + 1):
            pop_coords = [ind for ind, _ in population]
            scored_pop = self.evaluate_population(pop_coords)
            # Re-attach labels
            scored_with_labels = [(scored_pop[i][0], scored_pop[i][1], population[i][1]) for i in
                                  range(len(population))]
            population = self.selection(scored_with_labels, gen)
            self.post_generation_hook([ind for ind, _ in population], gen)

            # Save state at the end of each generation (or every n generations)
            if gen != 0 and gen % 5 == 0 or gen == self.generations:
                self.save_state(population, gen)

        self.timing["GA Loop"] = time.perf_counter() - t1
        return self.finalize([ind for ind, _ in population])

    def selection(
            self,
            scored_pop: List[Tuple[float, InternalCoordinatesList, str]],
            gen: int
    ) -> List[Tuple[InternalCoordinatesList, str]]:
        scored_pop.sort(key=lambda x: x[0])
        # Track survival
        elite_size = max(1, int(self.pop_size * 0.2))
        for i in range(min(len(scored_pop), elite_size)):
            origin = scored_pop[i][2]
            self.success_stats[origin][0] += 1  # Survived to elite

        for _, _, origin in scored_pop:
            self.success_stats[origin][1] += 1  # Total produced

        best_individuals = [(ind, origin) for _, ind, origin in scored_pop[:elite_size]]
        return self.reproduce(best_individuals, self.pop_size)

    def post_generation_hook(self, population: List[InternalCoordinatesList], gen: int):
        pass

    def finalize(self, population: List[InternalCoordinatesList]) -> MoleculeList:
        t2 = time.perf_counter()
        # Final evaluation of the last population
        self.evaluate_population(population)
        # Keep Cartesian optimization results; torsions cannot encode relaxed
        # bond lengths and angles.
        final_conformers = _to_molecule_list(self.evaluated_molecules)

        final_ranked = prune_conformers(final_conformers, 0.1)

        self.timing["Finalization"] = time.perf_counter() - t2

        self.node_runner.info("\n--- GA Success Rates (Elite Survival) ---")
        for origin, counts in self.success_stats.items():
            rate = (counts[0] / counts[1] * 100) if counts[1] > 0 else 0
            self.node_runner.info(f"{origin:<15}: {rate:6.2f}% ({counts[0]}/{counts[1]})")

        self.smart_optimizer.report()

        output_prefix = f"{self.__class__.__name__.lower()}_diversity"
        plot_conformer_diversity(final_ranked, output_prefix=output_prefix)

        if self.profile:
            self.node_runner.info(f"\n--- {self.__class__.__name__} CPU Profile ---")
            for k, v in self.timing.items(): self.node_runner.info(f"{k:<25}: {v:.4f}s")
        return _to_molecule_list(list(final_ranked)[:self.num_confs])


class StandardGA(BaseGA):
    """Score generated geometries; optimize only the final population."""

    def finalize(self, population: List[InternalCoordinatesList]) -> MoleculeList:
        molecules = [self.molecule_from_coordinates(ind) for ind in population]
        optimized = self.evaluate_molecules(molecules, optimize=True)
        if optimized:
            self.best_energy_seen = min(m.properties["energy"] for m in optimized)
        ranked = prune_conformers(_to_molecule_list(optimized), 0.1)
        return _to_molecule_list(list(ranked)[:self.num_confs])


class MinimizingGA(BaseGA):
    """Relax every generation through the selected molecule evaluator."""

    def evaluate_population(self, population: List[InternalCoordinatesList]) -> List[Tuple[float, InternalCoordinatesList]]:
        molecules = [self.molecule_from_coordinates(ind) for ind in population]
        if self.smart_opt:
            starting = self.evaluate_molecules(molecules)
            partial = self.evaluate_molecules(molecules, optimize=True,
                                              max_iters=max(1, self.max_iters // 10))
            keep = []
            for index, (start, current) in enumerate(zip(starting, partial)):
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
        self.evaluated_molecules = optimized
        if optimized:
            self.best_energy_seen = min(self.best_energy_seen,
                                       min(m.properties["energy"] for m in optimized))
        return [(mol.properties["energy"], self.coordinates_from_molecule(mol))
                for mol in optimized]


class DiversityGA(BaseGA):
    def __init__(self, *args, n_prune: int = 10, prune_rms_thresh: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_prune = n_prune
        self.prune_rms_thresh = prune_rms_thresh
        self.rmsd_history = []
        if n_prune < 1:
            raise ValueError("n_prune must be positive")

    def selection(
            self,
            scored_pop: List[Tuple[float, InternalCoordinatesList, str]],
            gen: int
    ) -> List[Tuple[InternalCoordinatesList, str]]:
        if not scored_pop:
            raise RuntimeError("DiversityGA.selection received an empty population.")

        # Track survival to elite (top 20%)
        scored_pop.sort(key=lambda x: x[0])
        elite_size = min(len(scored_pop), max(1, int(self.pop_size * 0.2)))
        for i in range(elite_size):
            origin = scored_pop[i][2]
            self.success_stats[origin][0] += 1

        for _, _, origin in scored_pop:
            self.success_stats[origin][1] += 1

        # 1. Keep the low-energy candidates according to the selected backend
        min_energy = min(scored_pop, key=lambda x: x[0])[0]
        threshold = min_energy + 0.1 * abs(min_energy)
        accepted = [ind for eng, ind, origin in scored_pop if eng <= threshold]
        if not accepted:
            accepted = [min(scored_pop, key=lambda x: x[0])[1]]

        # 2. Generate a new population with 2x the size of the accepted population
        doubled_population = self.reproduce([(ind, "ga-select") for ind in accepted], 2 * self.pop_size)
        doubled_population = [ind for ind, origin in doubled_population]

        # 3. Score the doubled population for diversity
        doubled_arr = np.array([
            [c.get_actual_value(c.value) for c in ind.elements]
            for ind in doubled_population
        ])
        diversity_scores = self.get_pairwise_rmsd_scores(doubled_arr)

        # 4. Sort by diversity and take top 50%
        # We use a greedy MaxMin approach to select a diverse subset.
        selected_indices = [0]  # Start with the first one (arbitrary)
        remaining_indices = list(range(1, len(doubled_population)))
        target_size = int(len(doubled_population) * 0.5)

        # Pre-calculate full distance matrix for the doubled population
        data = np.deg2rad(doubled_arr)
        diffs = data[:, np.newaxis, :] - data[np.newaxis, :, :]
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        dist_matrix = np.sqrt(np.mean(np.square(np.rad2deg(diffs)), axis=-1))

        while len(selected_indices) < target_size:
            # For each remaining individual, find its distance to the closest selected individual
            min_dists = np.min(dist_matrix[remaining_indices][:, selected_indices], axis=1)
            # Pick the one that has the largest such minimum distance
            best_idx_in_remaining = np.argmax(min_dists)
            selected_indices.append(remaining_indices.pop(best_idx_in_remaining))

        selected_individuals = [doubled_population[i] for i in selected_indices]

        # 5. Attach "ga-select" origin label to selected individuals
        new_population = [(ind, "ga-select") for ind in selected_individuals]

        # 6. Statistics for plotting
        new_pop_coords = [ind for ind, origin in new_population]
        new_pop_arr = np.array([
            [c.get_actual_value(c.value) for c in ind.elements]
            for ind in new_pop_coords
        ])
        after_scores = self.get_pairwise_rmsd_scores(new_pop_arr)
        self.rmsd_history.append({
            "gen": gen,
            "high_before": float(np.max(diversity_scores)) if len(diversity_scores) > 0 else 0.0,
            "low_before": float(np.min(diversity_scores)) if len(diversity_scores) > 0 else 0.0,
            "high_after": float(np.max(after_scores)) if len(after_scores) > 0 else 0.0,
            "low_after": float(np.min(after_scores)) if len(after_scores) > 0 else 0.0
        })

        output = (
            f"Gen {gen:3d} | "
            f"RMSD before [low={np.min(diversity_scores) if len(diversity_scores) > 0 else 0:8.4f}, high={np.max(diversity_scores) if len(diversity_scores) > 0 else 0:8.4f}] | "
            f"after [low={np.min(after_scores) if len(after_scores) > 0 else 0:8.4f}, high={np.max(after_scores) if len(after_scores) > 0 else 0:8.4f}] | "
            f"Min energy = {min_energy:10.4f}, Acc = {len(accepted):3d}"
        )
        self.node_runner.info(output)
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
            combined.sort(key=lambda m: m.properties["energy"])
            pruned = prune_conformers(_to_molecule_list(combined), self.prune_rms_thresh)
            self.best_stored = _to_molecule_list(list(pruned)[:self.num_confs])
            self.write_conformers_xyz(self.best_stored, gen)

    def plot_rmsd_evolution(self, gen: int):
        import matplotlib.pyplot as plt
        import pandas as pd
        h = self.rmsd_history

        # Save RMSD evolution to CSV
        rmsd_evol_df = pd.DataFrame(h)
        rmsd_evol_df.to_csv("rmsd_evolution.csv", index=False)
        self.node_runner.info(" RMSD")

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
        self.node_runner.info(f" PCA: {pca_csv}")

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

    def finalize(self, population: List[InternalCoordinatesList]) -> MoleculeList:
        if not self.best_stored or self.generations % self.n_prune != 0:
            self.post_generation_hook(population, self.generations, force_prune=True)

        output_prefix = f"{self.__class__.__name__.lower()}"
        plot_conformer_diversity(self.best_stored, output_prefix=output_prefix)

        # Success stats and reporting (shared with BaseGA.finalize in intent)
        self.node_runner.info("\n--- GA Success Rates (Elite Survival) ---")
        for origin, counts in self.success_stats.items():
            rate = (counts[0] / counts[1] * 100) if counts[1] > 0 else 0
            self.node_runner.info(f"{origin:<15}: {rate:6.2f}% ({counts[0]}/{counts[1]})")
        self.smart_optimizer.report()

        if self.profile:
            self.node_runner.info(f"\n--- {self.__class__.__name__} CPU Profile ---")
            for k, v in self.timing.items(): self.node_runner.info(f"{k:<25}: {v:.4f}s")

        self.node_runner.info("")
        return self.best_stored


def generate_ga_conformers(initial_mol: Molecule, **kwargs):
    return StandardGA(initial_mol, **kwargs).run()


def generate_ga_min_conformers(initial_mol: Molecule, **kwargs):
    return MinimizingGA(initial_mol, **kwargs).run()


def generate_ga_select_conformers(initial_mol: Molecule, **kwargs):
    return DiversityGA(initial_mol, **kwargs).run()
