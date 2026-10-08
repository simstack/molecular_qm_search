from __future__ import annotations
import asyncio
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
            prune_rms_thresh: float = 0.1,
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
        # [elite survivors, produced] for each population-change label.
        self.success_stats: Dict[str, List[int]] = {}

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

    def save_state(self, population, gen: int):
        state = {
            "population": population,
            "gen": gen,
            "best_stored": self.best_stored,
            "success_stats": self.success_stats,
            "energy_history": self.energy_history,
            "best_energy_seen": self.best_energy_seen,
            "random_state": self.population_generator.random.getstate()
        }
        if hasattr(self, "rmsd_history"):
            state["rmsd_history"] = self.rmsd_history
        if hasattr(self, "relaxed"):
            state["relaxed"] = [self.relaxed.get(id(coords)) for coords, _, _ in population]

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
        population = state.get("population")
        if not population or any(not isinstance(item, tuple) or len(item) != 3 for item in population):
            raise ValueError("GA checkpoint population has no energies")
        self.best_stored = state.get("best_stored", MoleculeList())
        self.success_stats = state["success_stats"]
        self.energy_history = state["energy_history"]
        self.best_energy_seen = state.get("best_energy_seen", float('inf'))
        if hasattr(self, "rmsd_history") and "rmsd_history" in state:
            self.rmsd_history = state["rmsd_history"]
        if hasattr(self, "relaxed"):
            relaxed = state.get("relaxed")
            if relaxed is None or len(relaxed) != len(population):
                raise ValueError("GA checkpoint is missing relaxed geometries")
            self.relaxed = {}
            for (coords, _, _), molecule in zip(population, relaxed):
                if molecule is not None:
                    self.relaxed[id(coords)] = molecule
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
        self.timing["Setup"] = time.perf_counter() - t0

    def evaluate_population(self, population: List[InternalCoordinatesList]) -> List[Tuple[float, InternalCoordinatesList]]:
        molecules = [self.molecule_from_coordinates(ind) for ind in population]
        self.evaluated_molecules = self.evaluate_molecules(molecules)
        return [(mol.properties["energy"], ind)
                for mol, ind in zip(self.evaluated_molecules, population)]

    def reproduce(self, best_individuals, target_size):
        return self.population_generator.reproduce(
            best_individuals, target_size, self.prune_rms_thresh,
        )

    def dihedral_rmsd(self, ind1: InternalCoordinatesList, ind2: InternalCoordinatesList) -> float:
        return dihedral_rmsd(ind1, ind2)

    def _report(self, message: str) -> None:
        """Send a message to the task log and the node runner info log."""
        if hasattr(self.node_runner, "log_string"):
            self.node_runner.log(message)
        self.node_runner.info(message)

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

    def _ranked_population(self, molecules: List[Molecule]) -> MoleculeList:
        if not molecules:
            raise ValueError("Final population is empty")
        for molecule in molecules:
            energy = molecule.properties.get("energy")
            if energy is None or not np.isfinite(energy):
                raise ValueError("Population molecule has no finite energy")
        ranked = sorted(molecules, key=lambda molecule: float(molecule.properties["energy"]))
        kept = []
        kept_coords = []
        for molecule in ranked:
            coords = self.coordinates_from_molecule(molecule)
            if any(self.dihedral_rmsd(coords, previous) < self.prune_rms_thresh for previous in kept_coords):
                continue
            kept.append(molecule)
            kept_coords.append(coords)
            if len(kept) == self.num_confs:
                break
        if not kept:
            raise ValueError("Final population is empty")
        return _to_molecule_list(kept)

    def _unique_conformers(self, scored_pop):
        """Keep the lowest-energy conformer of each dihedral neighborhood."""
        kept = []
        for energy, coords, origin in scored_pop:
            if any(self.dihedral_rmsd(coords, previous) < self.prune_rms_thresh
                   for _, previous, _ in kept):
                continue
            kept.append((energy, coords, origin))
        if not kept:
            raise ValueError("Uniqueness filter removed every conformer")
        return kept

    def _breed(self, carried, gen: int):
        requested = self.pop_size
        population = self.reproduce(carried, requested)
        if len(population) < requested:
            rejected = self.population_generator.rejected_rmsd
            if rejected is None or not np.isfinite(rejected):
                raise ValueError("Population shrank without a rejected dihedral RMSD")
            self.pop_size = len(population)
            self._report(
                f"Gen {gen}: kept {len(population)} of {requested} conformers; "
                f"the next candidate dihedral RMSD was {rejected:.4f} degrees, "
                f"below prune_rms_thresh {self.prune_rms_thresh}. "
                f"Population size is now {self.pop_size}."
            )
        return population

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
        start = 0
        for index in range(len(spans) - 1):
            previous, current = spans[index], spans[index + 1]
            if previous == 0 and current == 0:
                continue
            if previous == 0 or current == 0 or max(previous, current) / min(previous, current) > 2:
                start = index + 1
        if start and start != self._energy_plot_start:
            self._report(
                f"Dropped {start} initial iterations from the energy plot; "
                "the min-max energy range changed by more than a factor of 2."
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
        for name in ("energy_chart", "energy_histogram", "diversity_chart"):
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
        self._publish_energy_histogram([
            float(molecule.properties["energy"]) for molecule in molecules
        ])
        table = SimpleTable(name="Population change success rates")
        table.add_column("change", SimpleTableColumnType.STRING)
        table.add_column("produced", SimpleTableColumnType.NUMBER)
        table.add_column("elite_survivors", SimpleTableColumnType.NUMBER)
        table.add_column("success_rate", SimpleTableColumnType.NUMBER)
        self._report("\n--- GA Success Rates (Elite Survival) ---")
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
        self._persist_live_artifacts()
        self._report(f"Returning {len(molecules)} population molecules.")

    def run(self) -> MoleculeList:
        self.setup()
        self._report("\n--- GA Parameters ---")
        self._report(f"Atoms:          {len(self.initial_mol.atoms)}")
        self._report(f"Population:     {self.pop_size}")
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
        self._report(f"Rotatable Bonds: {len(self.template_coords.elements)}")
        self._report(f"Dihedrals:       {self.dihedrals}")
        self._report(f"Dihedral Types:  {self.dihedral_types}")
        self._report("---------------------\n")

        if not self.template_coords.elements:
            self._report("No rotatable bonds found.")
            evaluated = self.evaluate_molecules([self.initial_mol], optimize=True)
            self._record_energy_iteration(0, [mol.properties["energy"] for mol in evaluated])
            molecules = self._ranked_population(evaluated)
            self._publish_run_report(list(molecules))
            return molecules

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
            pending = [index for index, (_, _, energy) in enumerate(population) if energy is None]
            if pending:
                scored_pop = self.evaluate_population([population[index][0] for index in pending])
                if len(scored_pop) != len(pending):
                    raise ValueError(
                        f"Evaluator returned {len(scored_pop)} energies for {len(pending)} individuals"
                    )
                for index, (energy, coords) in zip(pending, scored_pop):
                    if energy is None or not np.isfinite(energy):
                        raise ValueError("Population conformer has no finite energy")
                    population[index] = (coords, population[index][1], float(energy))
            energies = []
            scored_coords = []
            for coords, _, energy in population:
                if energy is None or not np.isfinite(energy):
                    raise ValueError("Population conformer has no finite energy")
                energies.append(energy)
                scored_coords.append(coords)
            self._record_energy_iteration(gen, energies)
            self._publish_energy_chart()
            self._publish_energy_histogram(energies)
            self._publish_diversity_chart(scored_coords)
            self._persist_live_artifacts()
            scored_with_labels = [(energy, coords, origin) for coords, origin, energy in population]
            population = self.selection(scored_with_labels, gen)
            self.post_generation_hook([coords for coords, _, _ in population], gen)

            # Save state at the end of each generation (or every n generations)
            if gen != 0 and gen % 5 == 0 or gen == self.generations:
                self.save_state(population, gen)

        self.timing["GA Loop"] = time.perf_counter() - t1
        molecules = self.finalize(population)
        returned = list(molecules)
        if not 1 <= len(returned) <= self.num_confs:
            raise ValueError(
                f"Expected between 1 and {self.num_confs} population molecules, got {len(returned)}"
            )
        self._publish_run_report(returned)
        return molecules

    def selection(
            self,
            scored_pop: List[Tuple[float, InternalCoordinatesList, str]],
            gen: int
    ):
        scored_pop.sort(key=lambda x: x[0])
        elite_size = max(1, int(self.pop_size * 0.2))
        elite_count = min(len(scored_pop), elite_size)
        for index, (_, _, origin) in enumerate(scored_pop):
            self._count_origin(origin, elite=index < elite_count)

        carried = [
            (coords, origin, energy)
            for energy, coords, origin in self._unique_conformers(scored_pop)[:elite_size]
        ]
        return self._breed(carried, gen)

    def post_generation_hook(self, population: List[InternalCoordinatesList], gen: int):
        pass

    def finalize(self, population) -> MoleculeList:
        t2 = time.perf_counter()
        self.evaluate_population([coords for coords, _, _ in population])
        final_ranked = self._ranked_population(self.evaluated_molecules)

        self.timing["Finalization"] = time.perf_counter() - t2

        self.smart_optimizer.report()

        output_prefix = f"{self.__class__.__name__.lower()}_diversity"
        plot_conformer_diversity(final_ranked, output_prefix=output_prefix)

        if self.profile:
            self._report(f"\n--- {self.__class__.__name__} CPU Profile ---")
            for k, v in self.timing.items(): self._report(f"{k:<25}: {v:.4f}s")
        return final_ranked


class StandardGA(BaseGA):
    """Score generated geometries; optimize only the final population."""

    def finalize(self, population) -> MoleculeList:
        molecules = [self.molecule_from_coordinates(coords) for coords, _, _ in population]
        optimized = self.evaluate_molecules(molecules, optimize=True)
        if optimized:
            self.best_energy_seen = min(m.properties["energy"] for m in optimized)
        return self._ranked_population(optimized)


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
        paired = []
        for molecule in optimized:
            coords = self.coordinates_from_molecule(molecule)
            self.relaxed[id(coords)] = molecule
            paired.append((float(molecule.properties["energy"]), coords))
        return paired

    def finalize(self, population) -> MoleculeList:
        pending = [index for index, (_, _, energy) in enumerate(population) if energy is None]
        if pending:
            scored = self.evaluate_population([population[index][0] for index in pending])
            if len(scored) != len(pending):
                raise ValueError(
                    f"Evaluator returned {len(scored)} energies for {len(pending)} individuals"
                )
            for index, (energy, coords) in zip(pending, scored):
                population[index] = (coords, population[index][1], float(energy))
        molecules = []
        for coords, _, _ in population:
            molecule = self.relaxed.get(id(coords))
            if molecule is None:
                raise ValueError("Minimizing GA is missing the relaxed geometry for a carried conformer")
            molecules.append(molecule)
        return self._ranked_population(molecules)


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
        elite_size = min(len(scored_pop), max(1, int(self.pop_size * 0.2)))
        for index, (_, _, origin) in enumerate(scored_pop):
            self._count_origin(origin, elite=index < elite_size)

        min_energy = scored_pop[0][0]
        threshold = min_energy + 0.1 * abs(min_energy)
        accepted = [item for item in scored_pop if item[0] <= threshold]
        if not accepted:
            accepted = [scored_pop[0]]
        unique = self._unique_conformers(accepted)
        if len(unique) > self.pop_size:
            selected_indices = [0]
            remaining = list(range(1, len(unique)))
            rejected_rmsd = None
            while remaining and len(selected_indices) < self.pop_size:
                distances = [
                    min(
                        self.dihedral_rmsd(unique[candidate][1], unique[chosen][1])
                        for chosen in selected_indices
                    )
                    for candidate in remaining
                ]
                best = int(np.argmax(distances))
                if distances[best] < self.prune_rms_thresh:
                    rejected_rmsd = float(distances[best])
                    break
                selected_indices.append(remaining.pop(best))
            selected = [unique[index] for index in selected_indices]
            if len(selected) < self.pop_size:
                if rejected_rmsd is None or not np.isfinite(rejected_rmsd):
                    raise ValueError("Population shrank without a rejected dihedral RMSD")
                requested = self.pop_size
                self.pop_size = len(selected)
                self._report(
                    f"Gen {gen}: kept {len(selected)} of {requested} conformers; "
                    f"the next candidate dihedral RMSD was {rejected_rmsd:.4f} degrees, "
                    f"below prune_rms_thresh {self.prune_rms_thresh}. "
                    f"Population size is now {self.pop_size}."
                )
            new_population = [(coords, origin, energy) for energy, coords, origin in selected]
        else:
            carried = [(coords, origin, energy) for energy, coords, origin in unique]
            new_population = self._breed(carried, gen)

        before_scores = self.get_pairwise_rmsd_scores([coords for _, coords, _ in accepted])
        after_scores = self.get_pairwise_rmsd_scores([coords for coords, _, _ in new_population])
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
            self.best_stored = self._ranked_population(combined)
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
                [coords for coords, _, _ in population], self.generations, force_prune=True,
            )

        output_prefix = f"{self.__class__.__name__.lower()}"
        plot_conformer_diversity(self.best_stored, output_prefix=output_prefix)

        self.smart_optimizer.report()

        if self.profile:
            self._report(f"\n--- {self.__class__.__name__} CPU Profile ---")
            for k, v in self.timing.items(): self._report(f"{k:<25}: {v:.4f}s")

        self._report("")
        return self.best_stored


def generate_ga_conformers(initial_mol: Molecule, **kwargs):
    return StandardGA(initial_mol, **kwargs).run()


def generate_ga_min_conformers(initial_mol: Molecule, **kwargs):
    return MinimizingGA(initial_mol, **kwargs).run()


def generate_ga_select_conformers(initial_mol: Molecule, **kwargs):
    return DiversityGA(initial_mol, **kwargs).run()
