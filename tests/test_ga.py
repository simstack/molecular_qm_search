import copy
import logging
import random

import pytest

from molecular_qm_models import Atom, Molecule, InternalCoordinatesList, InternalDihedralCoordinate
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType
from molecular_qm_search.optimization.ga_population import PopulationGenerator
from molecular_qm_search.optimization.ga_method import StandardGA, MinimizingGA, DiversityGA
from molecular_qm_search.optimization.lib.ga_evaluation import RDKitEvaluator, validate_results


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


class Evaluator:
    """A geometry-changing backend with an energy independent of the torsion."""
    def __init__(self):
        self.calls = []

    def score(self, molecules):
        self.calls.append(("score", len(molecules)))
        results = [Molecule.from_molecule(m) for m in molecules]
        for molecule in results:
            molecule.properties["energy"] = 5.0
        return results

    def optimize(self, molecules, *, max_iters):
        self.calls.append(("optimize", len(molecules), max_iters))
        results = [Molecule.from_molecule(m) for m in molecules]
        for molecule in results:
            molecule.atoms[0].y = 1.01
            molecule.properties.update(energy=-2.0, energy_unit="kcal/mol", method="test")
        return results


def test_rotatable_bond_discovery_receives_its_options(molecule, monkeypatch):
    captured = {}

    def fake_get_rotatable_bonds(mol, match_double_bonds=False, min_value=-180.0, max_value=180.0):
        captured["molecule"] = mol
        captured["match_double_bonds"] = match_double_bonds
        captured["min_value"] = min_value
        captured["max_value"] = max_value
        return InternalCoordinatesList()

    import molecular_qm_util
    monkeypatch.setattr(
        molecular_qm_util, "get_rotatable_bonds", fake_get_rotatable_bonds, raising=False,
    )
    PopulationGenerator(
        molecule, match_double_bonds=False, rotatable_bond_min=-90, rotatable_bond_max=120,
    )
    assert captured["molecule"] is molecule
    assert captured["match_double_bonds"] is False
    assert captured["min_value"] == -90
    assert captured["max_value"] == 120


def test_inverted_rotatable_bond_bounds_are_rejected(molecule, coordinates):
    with pytest.raises(ValueError, match="rotatable_bond_max"):
        PopulationGenerator(molecule, coordinates=coordinates, rotatable_bond_min=10, rotatable_bond_max=10)


def test_generation_is_reproducible_and_does_not_modify_inputs(molecule, coordinates):
    original_molecule, original_coordinates = copy.deepcopy((molecule, coordinates))
    state = random.getstate()
    first = PopulationGenerator(molecule, coordinates=coordinates, seed=5)
    second = PopulationGenerator(molecule, coordinates=coordinates, seed=5)
    pop1, pop2 = first.generate(5), second.generate(5)
    assert [p.elements[0].value for p, *_ in pop1] == [p.elements[0].value for p, *_ in pop2]
    before = copy.deepcopy(pop1[1][0])
    result = first.molecule_from_coordinates(pop1[1][0])
    assert result is not molecule
    assert pop1[1][0].model_dump() == before.model_dump()
    assert molecule.model_dump() == original_molecule.model_dump()
    assert coordinates.model_dump() == original_coordinates.model_dump()
    assert random.getstate() == state


def test_each_dihedral_gets_a_full_circle_histogram(molecule, coordinates):
    runner = logging.getLogger("dihedral-histograms")
    ga = StandardGA(
        molecule, runner, evaluator=Evaluator(), coordinates=coordinates,
        pop_size=4, generations=0, num_confs=4,
    )
    ga.setup()
    population = [coords for coords, *_ in ga.population_generator.generate(8)]
    targets = [-180.0, -150.0, -90.0, 0.0, 30.0, 149.9, 150.0, 180.0]
    for individual, angle in zip(population, targets):
        coordinate = individual.elements[0]
        low = coordinate.min_values[0]
        high = coordinate.max_values[0]
        coordinate.value = (angle - low) / (high - low)
        coordinate.real_values = [angle]
    ga._publish_dihedral_histograms(population)
    chart = runner.dihedral_histogram_0
    assert chart.title.text == "Dihedral 0-1-2-3"
    assert [row["angle"] for row in chart.data] == [
        f"{-180 + index * 30} – {-150 + index * 30}" for index in range(12)
    ]
    assert [row["count"] for row in chart.data] == [2, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    assert sum(row["count"] for row in chart.data) == 8
    assert getattr(runner, "dihedral_histogram_1", None) is None
    ga._publish_dihedral_histograms(population[:1])
    assert runner.dihedral_histogram_0 is chart
    assert sum(row["count"] for row in chart.data) == 1

    empty = StandardGA(
        molecule, logging.getLogger("no-dihedrals"), evaluator=Evaluator(),
        coordinates=InternalCoordinatesList(), pop_size=1, generations=0,
    )
    empty.setup()
    empty._publish_dihedral_histograms([InternalCoordinatesList()])
    assert empty._dihedral_histogram_names == []


def test_double_bond_mutation_flips_relative_to_parent(molecule, coordinates):
    coordinates.elements[0].bond_type = InternalCoordinateBondType.DOUBLE
    generator = PopulationGenerator(molecule, coordinates=coordinates, mutation_rate=1,
                                    dihedral_interval=0)
    generated = generator.generate(1)
    history = [{"op": "I", "index": 0}]
    parents = [(generated[0][0], generated[0][1], generated[0][2], 0)]
    offspring = generator.reproduce(parents, 2, 0.1, history)
    before = parents[0][0].elements[0].value
    after = offspring[1][0].elements[0].value
    assert (after - before) % 1 == pytest.approx(0.5)


@pytest.mark.parametrize("ga_class", [StandardGA, MinimizingGA, DiversityGA])
def test_modes_use_injected_backend_and_preserve_final_geometry(
    ga_class, molecule, coordinates, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    evaluator = Evaluator()
    ga = ga_class(molecule, logging.getLogger("test"), evaluator=evaluator,
                  coordinates=coordinates, pop_size=3, num_confs=2, generations=1,
                  mutation_rate=1.0)
    ga.setup()
    assert evaluator.calls == []
    result = ga.run()
    assert result
    if ga_class is StandardGA:
        assert evaluator.calls[0] == ("optimize", 1, ga.max_iters)
        assert all(call[0] != "optimize" for call in evaluator.calls[1:])
        assert any(call[0] == "score" for call in evaluator.calls)
        assert all(m.properties["energy"] == 5.0 for m in result)
    else:
        assert any(call[0] == "optimize" for call in evaluator.calls)
        assert all(m.atoms[0].y == 1.01 for m in result)
        assert all(m.properties["energy"] == -2 for m in result)
        assert all(m.properties["method"] == "test" for m in result)
        if ga_class is not MinimizingGA:
            assert any(call[0] == "score" for call in evaluator.calls)


def test_rigid_molecule_uses_backend_without_rdkit(molecule, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = Evaluator()
    ga = StandardGA(molecule, logging.getLogger("test"), evaluator=evaluator,
                    coordinates=InternalCoordinatesList())
    result = ga.run()
    assert len(result) == 1
    assert evaluator.calls == [("optimize", 1, 500), ("score", 1)]
    assert result[0].atoms[0].y == 1.01


def test_smart_optimization_preserves_order_and_partial_results(molecule, coordinates):
    evaluator = Evaluator()
    ga = MinimizingGA(molecule, logging.getLogger("test"), evaluator=evaluator,
                      coordinates=coordinates, smart_opt=True, max_iters=100)
    ga.setup()
    decisions = iter([True, False, True])
    ga.smart_optimizer.should_discard = lambda *args: next(decisions)
    population = [ind for ind, *_ in ga.population_generator.generate(3)]
    assert len(ga.evaluate_population(population)) == 3
    assert evaluator.calls == [("score", 3), ("optimize", 3, 10), ("optimize", 1, 100)]
    assert len(ga.evaluated_molecules) == 3


@pytest.mark.parametrize("energy", [None, float("nan"), float("inf")])
def test_backend_invalid_energy_is_rejected(molecule, energy):
    molecule.properties["energy"] = energy
    with pytest.raises(ValueError, match="finite energy"):
        validate_results([molecule], [molecule])


def test_backend_missing_results_are_rejected(molecule):
    with pytest.raises(ValueError, match="one molecule per input"):
        validate_results([molecule], [])


class RecordingRunner:
    def __init__(self):
        self.log_string = ""
        self.info_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def log(self, message):
        self.log_string += f"{message}\n"

    def error(self, message):
        raise AssertionError(message)


class DistinctGeometryEvaluator:
    def score(self, molecules):
        results = []
        for index, molecule in enumerate(molecules):
            result = Molecule.from_molecule(molecule)
            result.atoms[0].x = float(index) * 2.0
            result.properties["energy"] = float(index)
            results.append(result)
        return results

    def optimize(self, molecules, *, max_iters):
        return self.score(molecules)


def test_run_returns_conformers_chart_table_and_task_log(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=DistinctGeometryEvaluator(), coordinates=coordinates,
        pop_size=12, num_confs=6, generations=2, mutation_rate=1.0, seed=1,
        prune_rms_thresh=0.1, heavy_atom_rmsd_thresh=1e-6,
    )
    conformers = ga.run()
    assert len(conformers) == 12
    assert [float(molecule.properties["energy"]) for molecule in conformers] == [
        row["energy"] for row in runner.conformation_energies.row
    ]
    assert "Population:     12" in runner.log_string
    assert "Seed population: the minimized input geometry alone" in runner.log_string
    assert "Minimized the input geometry" in runner.log_string
    assert "at or above prune_rms_thresh" in runner.log_string
    assert "Dihedral Types:" in runner.log_string
    assert runner.log_string == "\n".join(runner.info_messages) + "\n"
    chart = runner.energy_chart
    assert [series.yKey for series in chart.series] == [
        "min-energy", "rank-1/4", "rank-2/4", "rank-3/4", "rank-4/4", "max-energy",
    ]
    assert [series.title for series in chart.series] == [
        "min-energy", "rank 1/4", "rank 2/4", "rank 3/4", "rank 4/4", "max-energy",
    ]
    assert [row["iteration"] for row in chart.data] == [0, 1, 2]
    assert all(
        row["min-energy"] <= row["rank-1/4"] <= row["rank-2/4"]
        <= row["rank-3/4"] <= row["rank-4/4"] <= row["max-energy"]
        for row in chart.data
    )
    accepted = runner.accepted_histogram
    assert accepted.title.text == "Newly accepted conformers"
    assert [series.yKey for series in accepted.series] == ["accepted"]
    assert [row["iteration"] for row in accepted.data] == [0, 1, 2]
    assert accepted.data[0]["accepted"] >= 1
    assert all(row["accepted"] >= 0 for row in accepted.data)
    assert sum(row["accepted"] for row in accepted.data) >= len(runner.conformation_energies.row)
    table = runner.operator_stats
    assert table.heading == ["change", "produced", "elite_survivors", "success_rate"]
    changes = [row["change"] for row in table.row]
    assert "initial" in changes
    assert "copy" not in changes
    assert any(change.startswith("mutation-") for change in changes)
    for row in table.row:
        assert 0 <= row["success_rate"] <= 1
        assert row["elite_survivors"] <= row["produced"]
    energies = runner.conformation_energies
    assert energies.heading == ["conformation", "origin", "energy", "history"]
    assert energies.type == ["number", "string", "number", "string"]
    assert energies.row
    for row in energies.row:
        assert row["history"].endswith("I,0")
        assert row["history"].split(" < ")[1].split(",")[0] in {"I", "M", "X"}
    assert [row["energy"] for row in energies.row] == sorted(row["energy"] for row in energies.row)
    histogram = runner.energy_histogram
    assert histogram.series[0].type == "column"
    assert histogram.series[0].yKey == "count"
    rows = histogram.options["conformations"]
    assert [row["energy"] for row in rows] == sorted(row["energy"] for row in rows)
    assert [row["conformation"] for row in rows] == list(range(1, len(rows) + 1))
    assert sum(row["count"] for row in histogram.data) == len(rows)
    assert chart.data[-1]["min-energy"] == rows[0]["energy"]
    assert chart.data[-1]["max-energy"] == rows[-1]["energy"]
    assert chart.options["conformations"] == rows
    assert runner.diversity_chart.series[0].type == "scatter"
    assert len(runner.diversity_chart.data) == 12
    assert {key for row in runner.diversity_chart.data for key in row} == {"pc1", "pc2"}


def test_success_rate_counts_a_rejected_child_once(molecule, coordinates):
    ga = StandardGA(
        molecule, logging.getLogger("test"), evaluator=Evaluator(), coordinates=coordinates,
        pop_size=1, generations=0, mutation_rate=1.0, dihedral_interval=90, seed=1,
        prune_rms_thresh=0.1,
    )
    ga.setup()
    ga._history.append({"op": "I", "index": 0})
    seed = (copy.deepcopy(ga.template_coords), "initial", 1.0, 0)
    kept = ga._select_next_population([(1.0, seed[0], "initial", 0)], 0)
    assert ga.success_stats["initial"] == [1, 1]
    mutation = next(origin for origin in ga.success_stats if origin.startswith("mutation-"))
    assert ga.success_stats[mutation] == [0, 1]
    ga._select_next_population([
        (energy, coords, origin, history_index)
        for coords, origin, energy, history_index in kept
    ], 1)
    assert ga.success_stats["initial"] == [1, 1]
    assert ga.success_stats[mutation] == [0, 2]


def test_charts_refresh_on_every_iteration(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RecordingRunner()
    seen = []

    class WatchingEvaluator(DistinctGeometryEvaluator):
        def score(self, molecules):
            chart = getattr(runner, "energy_chart", None)
            if chart is not None:
                diversity = getattr(runner, "diversity_chart", None)
                seen.append({
                    "iterations": [row["iteration"] for row in chart.data],
                    "histogram": sum(row["count"] for row in runner.energy_histogram.data),
                    "diversity": None if diversity is None else len(diversity.data),
                })
            return super().score(molecules)

    ga = StandardGA(
        molecule, runner, evaluator=WatchingEvaluator(), coordinates=coordinates,
        pop_size=4, num_confs=4, generations=2, mutation_rate=1.0, seed=1,
        prune_rms_thresh=0.1, heavy_atom_rmsd_thresh=1e-6,
    )
    ga.run()
    assert seen == [
        {"iterations": [0], "histogram": 4, "diversity": 4},
        {"iterations": [0, 1], "histogram": 4, "diversity": 4},
    ]
    assert [row["iteration"] for row in runner.energy_chart.data] == [0, 1, 2]


def test_diversity_chart_separates_distinct_dihedrals(molecule, coordinates):
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=Evaluator(), coordinates=coordinates, pop_size=2, num_confs=2,
    )
    ga.setup()
    folded = copy.deepcopy(ga.template_coords)
    extended = copy.deepcopy(ga.template_coords)
    folded.elements[0].value = 0.0
    extended.elements[0].value = 0.5
    ga._publish_diversity_chart([folded, extended])
    points = runner.diversity_chart.data
    assert points[0]["pc1"] != pytest.approx(points[1]["pc1"])
    same = copy.deepcopy(folded)
    ga._publish_diversity_chart([folded, same])
    collapsed = runner.diversity_chart.data
    assert collapsed[0]["pc1"] == pytest.approx(0.0)
    assert collapsed[1]["pc1"] == pytest.approx(0.0)


def test_rank_quarters_follow_the_sorted_population(molecule, coordinates):
    ga = StandardGA(molecule, logging.getLogger("test"), coordinates=coordinates, pop_size=1)
    ga._record_energy_iteration(0, [4.0, 1.0, 3.0, 2.0, 5.0])
    point = ga.energy_history[0]
    assert point["min_energy"] == 1.0
    assert point["max_energy"] == 5.0
    assert point["rank-1/4"] == 2.0
    assert point["rank-2/4"] == 3.0
    assert point["rank-3/4"] == 4.0
    assert point["rank-4/4"] == 5.0
    ga._record_accepted_iteration(0, [0])
    ga._record_accepted_iteration(1, [0, 1, 2])
    ga._record_accepted_iteration(2, [2])
    assert [row["accepted"] for row in ga.accepted_history] == [1, 2, 0]


def test_energy_plot_drops_iterations_until_range_settles(molecule, coordinates):
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, coordinates=coordinates, pop_size=4, num_confs=4,
    )
    # 200 -> 8 exceeds a factor of 10 and would drop the first two iterations.
    # 7 -> 0.4 would drop everything before the last two; the plot still keeps 10.
    spans = [200.0, 180.0, 8.0] + [7.0] * 7 + [0.4, 0.4]
    ga.energy_history = [
        {
            "iteration": index,
            "min_energy": 0.0,
            "max_energy": span,
            "rank-1/4": span / 4,
            "rank-2/4": span / 2,
            "rank-3/4": 3 * span / 4,
            "rank-4/4": span,
        }
        for index, span in enumerate(spans)
    ]
    ga._publish_energy_chart()
    assert [row["iteration"] for row in runner.energy_chart.data] == list(range(2, len(spans)))
    assert "Dropped 2 initial iterations" in runner.log_string
    assert "factor of 10" in runner.log_string


def test_restart_restores_generator_random_state(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ga = StandardGA(molecule, logging.getLogger("test"), evaluator=Evaluator(),
                    coordinates=coordinates, mutation_rate=1.0)
    ga.setup()
    generated = ga.population_generator.generate(4)
    population = []
    for coords, origin, _energy in generated:
        index = len(ga._history)
        ga._history.append({"op": "I", "index": index})
        population.append((coords, origin, 1.0, index))
    ga.save_state(population, 2)
    expected = ga.reproduce(population[:1], 4)
    restored, generation = ga.load_state()
    actual = ga.reproduce(restored[:1], 4)
    assert generation == 2
    assert [p.elements[0].value for p, *_ in actual] == [p.elements[0].value for p, *_ in expected]


def test_restart_keeps_the_minimized_geometry(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = Evaluator()
    ga = StandardGA(
        molecule, logging.getLogger("test"), evaluator=evaluator, coordinates=coordinates,
        pop_size=2, generations=0, mutation_rate=1.0,
    )
    ga.run()
    assert evaluator.calls[0] == ("optimize", 1, 500)
    resumed = StandardGA(
        molecule, logging.getLogger("test"), evaluator=evaluator, coordinates=coordinates,
        pop_size=2, generations=1, mutation_rate=1.0, restart=True,
    )
    resumed.run()
    assert [call for call in evaluator.calls if call[0] == "optimize"] == [("optimize", 1, 500)]
    assert resumed.initial_mol.atoms[0].y == 1.01


def test_later_generations_score_only_new_conformers(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = Evaluator()
    ga = StandardGA(
        molecule, logging.getLogger("test"), evaluator=evaluator, coordinates=coordinates,
        pop_size=5, num_confs=5, generations=1, mutation_rate=1.0, dihedral_interval=40,
    )
    ga.run()
    score_sizes = [call[1] for call in evaluator.calls if call[0] == "score"]
    assert score_sizes == [1, 5, 5]


def test_noop_mutations_do_not_shrink_the_population_cap(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=Evaluator(), coordinates=coordinates,
        pop_size=4, num_confs=4, generations=0, mutation_rate=1.0, dihedral_interval=0,
    )
    result = list(ga.run())
    assert ga.pop_size == 4
    assert 1 <= len(result) <= 4
    assert "rejected 4 with no dihedral change" in runner.log_string
    assert "maximum pop_size 4" in runner.log_string
    assert "num_confs caps the returned list of ga-min and ga-select" in runner.log_string


def test_wide_rmsd_threshold_drops_conformers_and_says_so(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=Evaluator(), coordinates=coordinates,
        pop_size=4, num_confs=4, generations=0, mutation_rate=1.0,
        dihedral_interval=40, prune_rms_thresh=360, seed=1,
    )
    result = list(ga.run())
    assert len(result) == 1
    assert ga.pop_size == 4
    assert "below prune_rms_thresh 360" in runner.log_string
    assert "nearest" in runner.log_string
    assert "maximum pop_size 4" in runner.log_string


def test_lower_energy_child_replaces_a_close_parent(molecule, coordinates):
    ga = StandardGA(
        molecule, logging.getLogger("test"), evaluator=Evaluator(), coordinates=coordinates,
        pop_size=2, num_confs=1, mutation_rate=1.0, prune_rms_thresh=0.1,
    )
    ga.setup()
    close = copy.deepcopy(ga.template_coords)
    far = copy.deepcopy(ga.template_coords)
    ga.population_generator._shift(far.elements[0], 90)
    child = copy.deepcopy(close)

    def fake_reproduce(parents, target_size, prune_rms_thresh, history):
        ga.population_generator.reproduction_stats = {
            "accepted": 1,
            "rejected_rmsd": 0,
            "rejected_noop": 0,
            "closest_rejected_rmsd": None,
        }
        history.append({
            "op": "M", "index": 2, "ancestor": 0, "dihedral": 0, "angle": 30.0,
        })
        return list(parents) + [(child, "mutation-single", None, 2)]

    ga.population_generator.reproduce = fake_reproduce
    ga.evaluate_population = lambda population: [(0.1, population[0])]
    selected = ga.selection(
        [(5.0, close, "initial", 0), (1.0, far, "initial", 1)], 0,
    )
    assert ga.pop_size == 2
    assert [(origin, energy) for _, origin, energy, _index in selected] == [
        ("mutation-single", 0.1),
        ("initial", 1.0),
    ]


def test_num_confs_caps_only_the_returned_list(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=Evaluator(), coordinates=coordinates,
        pop_size=3, num_confs=1, generations=0, mutation_rate=1.0,
        dihedral_interval=40, seed=1,
    )
    seen = []
    ga.post_generation_hook = lambda population, gen: seen.append(len(population))
    result = list(ga.run())
    assert seen == [3]
    assert len(result) == 3
    assert ga.pop_size == 3
    assert "UFF relaxation is not applied" in runner.log_string
    assert "returning 3" in runner.log_string
    assert "Stopped at num_confs" not in runner.log_string


def test_relaxation_into_one_well_does_not_drop_the_population(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class SameGeometryEvaluator(Evaluator):
        def optimize(self, molecules, *, max_iters):
            self.calls.append(("optimize", len(molecules), max_iters))
            results = []
            for index, source in enumerate(molecules):
                result = Molecule.from_molecule(source)
                for atom, template in zip(result.atoms, molecule.atoms):
                    atom.x, atom.y, atom.z = template.x, template.y, template.z
                result.properties.update(energy=float(index), energy_unit="kcal/mol", method="test")
                results.append(result)
            return results

    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=SameGeometryEvaluator(), coordinates=coordinates,
        pop_size=3, num_confs=3, generations=0, mutation_rate=1.0,
        dihedral_interval=40, seed=1,
    )
    result = list(ga.run())
    assert len(result) == 3
    assert ga.evaluator.calls[0] == ("optimize", 1, 500)
    assert all(call[0] != "optimize" for call in ga.evaluator.calls[1:])
    assert "UFF relaxation is not applied" in runner.log_string
    assert "returning 3" in runner.log_string


def test_crossover_requires_two_parents(molecule, coordinates):
    generator = PopulationGenerator(molecule, coordinates=coordinates, mutation_rate=0)
    coords, origin, energy = generator.generate(1)[0]
    history = [{"op": "I", "index": 0}]
    with pytest.raises(ValueError, match="two parents"):
        generator.reproduce([(coords, origin, energy, 0)], 2, 0.1, history)


def test_history_walks_mutation_and_crossover_back_to_the_input(molecule, coordinates):
    generator = PopulationGenerator(
        molecule, coordinates=coordinates, mutation_rate=1.0,
        dihedral_interval=30.0, seed=1,
    )
    history = [{"op": "I", "index": 0}]
    coords, origin, energy = generator.generate(1)[0]
    mutated = generator.reproduce([(coords, origin, energy, 0)], 2, 0.1, history)
    mutation = history[mutated[1][3]]
    assert mutation["op"] == "M"
    assert mutation["ancestor"] == 0
    assert mutation["dihedral"] == 0
    assert mutation["angle"] != 0

    crosser = PopulationGenerator(
        molecule, coordinates=coordinates, mutation_rate=0.0, seed=1,
    )
    second = copy.deepcopy(coords)
    crosser._shift(second.elements[0], 40.0)
    parents = [
        (coords, "initial", 1.0, 0),
        (second, "mutation-single", 2.0, mutation["index"]),
    ]
    crossed = crosser.reproduce(parents, 3, 0.1, history)
    child_index = crossed[2][3]
    cross = history[child_index]
    assert cross["op"] == "X"
    assert {cross["a1"], cross["a2"]} == {0, mutation["index"]}
    assert cross["cut"] == 0

    ga = StandardGA(molecule, logging.getLogger("test"), coordinates=coordinates, pop_size=1)
    ga.setup()
    ga._history = history
    ga._remember_conformation_energies([(crossed[2][0], "crossover", 2.0, child_index)])
    trace = ga._conformation_rows[0]["history"]
    assert trace.startswith(f"{child_index} < X,{cross['a1']},{cross['a2']},{cross['cut']}")
    assert f"M,{mutation['ancestor']},{mutation['dihedral']},{mutation['angle']:+.6f}" in trace
    assert "I,0" in trace


def test_reproduction_rejects_a_parent_missing_from_history(molecule, coordinates):
    generator = PopulationGenerator(molecule, coordinates=coordinates, mutation_rate=1)
    coords, origin, energy = generator.generate(1)[0]
    with pytest.raises(ValueError, match="not in the GA history"):
        generator.reproduce([(coords, origin, energy, 4)], 2, 0.1, [{"op": "I", "index": 0}])


def test_non_positive_prune_threshold_is_rejected(molecule, coordinates):
    with pytest.raises(ValueError, match="prune_rms_thresh"):
        StandardGA(molecule, logging.getLogger("test"), coordinates=coordinates, prune_rms_thresh=0)
    with pytest.raises(ValueError, match="heavy_atom_rmsd_thresh"):
        StandardGA(
            molecule, logging.getLogger("test"), coordinates=coordinates,
            heavy_atom_rmsd_thresh=0,
        )
    with pytest.raises(ValueError, match="heavy_atom_rmsd_thresh"):
        StandardGA(
            molecule, logging.getLogger("test"), coordinates=coordinates,
            heavy_atom_rmsd_thresh=None,
        )


def test_hydrogen_rotation_is_pruned_by_heavy_atom_rmsd():
    molecule = Molecule(atoms=[
        Atom(element="C", x=0.0, y=1.2, z=0.0),
        Atom(element="C", x=0.0, y=0.0, z=0.0),
        Atom(element="C", x=1.50, y=0.0, z=0.0),
        Atom(element="H", x=2.05, y=0.90, z=0.0),
    ])
    torsion = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180, 180)
    torsion.compute(molecule)
    ga = StandardGA(
        molecule, logging.getLogger("test"),
        coordinates=InternalCoordinatesList(elements=[torsion]),
        pop_size=2, prune_rms_thresh=0.1, heavy_atom_rmsd_thresh=0.05,
    )
    ga.setup()
    turned = _offset(ga.template_coords, 90.0)
    start = ga.molecule_from_coordinates(ga.template_coords)
    rotated = ga.molecule_from_coordinates(turned)
    assert ga.dihedral_rmsd(ga.template_coords, turned) == pytest.approx(90.0, abs=1e-4)
    assert ga.heavy_atom_rmsd(start, rotated) == pytest.approx(0.0, abs=1e-6)
    kept, stats = ga._unique_conformers([
        (1.0, ga.template_coords, "initial", 0),
        (2.0, turned, "mutation-single", 1),
    ])
    assert [(origin, energy) for energy, _coords, origin, _index in kept] == [
        ("initial", 1.0),
    ]
    assert stats["dropped"] == 0
    assert stats["dropped_heavy"] == 1
    assert stats["nearest_dropped_heavy_rmsd"] == pytest.approx(0.0, abs=1e-6)

    rotated.properties["energy"] = 0.2
    start.properties["energy"] = 1.0
    ranked = list(ga._ranked_population([start, rotated], limit=None))
    assert len(ranked) == 1
    assert ranked[0].properties["energy"] == pytest.approx(0.2)

    hydrogens = Molecule(atoms=[
        Atom(element="H", x=0.0, y=0.0, z=0.0),
        Atom(element="H", x=0.74, y=0.0, z=0.0),
    ])
    with pytest.raises(ValueError, match="non-hydrogen"):
        ga.heavy_atom_rmsd(hydrogens, hydrogens)


def test_input_minimization_failure_stops_the_search(molecule):
    class DiscardingEvaluator(Evaluator):
        def optimize(self, molecules, *, max_iters):
            self.calls.append(("optimize", len(molecules), max_iters))
            result = Molecule.from_molecule(molecules[0])
            result.properties["energy"] = 1.0
            result.properties["force_field_discarded"] = True
            result.properties["force_field_gradient"] = 1.0
            return [result]

    ga = StandardGA(
        molecule, logging.getLogger("test"), evaluator=DiscardingEvaluator(),
        coordinates=InternalCoordinatesList(),
    )
    with pytest.raises(ValueError, match="high force-field gradient"):
        ga.run()


def test_checkpoint_without_energies_is_rejected(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ga = StandardGA(molecule, logging.getLogger("test"), evaluator=Evaluator(), coordinates=coordinates)
    ga.setup()
    population = [(coords, "initial") for coords, *_ in ga.population_generator.generate(2)]
    ga.save_state(population, 1)
    with pytest.raises(ValueError, match="no energies"):
        ga.load_state()


class CollapsingEvaluator:
    def score(self, molecules):
        return self.optimize(molecules, max_iters=1)

    def optimize(self, molecules, *, max_iters):
        template = Molecule.from_molecule(molecules[0])
        results = []
        for index, _molecule in enumerate(molecules):
            result = Molecule.from_molecule(template)
            result.properties["energy"] = float(index)
            result.properties["energy_unit"] = "kcal/mol"
            results.append(result)
        return results


def test_ga_min_keeps_one_conformer_from_one_basin(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ga = MinimizingGA(
        molecule, logging.getLogger("test"), evaluator=CollapsingEvaluator(), coordinates=coordinates,
        pop_size=3, num_confs=3, generations=0, mutation_rate=1.0, dihedral_interval=0,
    )
    assert len(list(ga.run())) == 1


def test_archive_drops_duplicate_dihedrals(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ga = DiversityGA(
        molecule, logging.getLogger("test"), evaluator=Evaluator(), coordinates=coordinates,
        pop_size=2, num_confs=2, n_prune=1, mutation_rate=1.0,
    )
    ga.setup()
    folded = copy.deepcopy(ga.template_coords)
    ga.post_generation_hook([folded, copy.deepcopy(folded)], 1)
    assert len(list(ga.best_stored)) == 1


def test_overlapping_conformer_is_excluded_and_not_scored(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = Evaluator()
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=evaluator, coordinates=coordinates,
        pop_size=2, num_confs=2, generations=0, mutation_rate=1.0,
    )
    ga.setup()
    assert ga._overlapping_atoms(molecule) is None
    overlapped = Molecule.from_molecule(molecule)
    overlapped.atoms[3].x = overlapped.atoms[0].x
    overlapped.atoms[3].y = overlapped.atoms[0].y
    overlapped.atoms[3].z = overlapped.atoms[0].z
    overlap = ga._overlapping_atoms(overlapped)
    assert overlap is not None
    assert "C2 and C4" in overlap
    original = ga.molecule_from_coordinates

    def collide(coords):
        built = original(coords)
        if coords is bad:
            built.atoms[3].x = built.atoms[0].x
            built.atoms[3].y = built.atoms[0].y
            built.atoms[3].z = built.atoms[0].z
        return built

    good = copy.deepcopy(ga.template_coords)
    bad = copy.deepcopy(ga.template_coords)
    ga.molecule_from_coordinates = collide
    kept = ga._exclude_overlaps([
        (good, "initial", None, 0), (bad, "mutation-single", None, 1),
    ])
    assert kept == [(good, "initial", None, 0)]
    assert evaluator.calls == []
    assert "positive infinity" in runner.log_string


def test_hydrogens_0_982_angstrom_apart_overlap_and_a_normal_contact_does_not():
    seed = Molecule(atoms=[
        Atom(element="C", x=0, y=0, z=0),
        Atom(element="H", x=1.09, y=0, z=0),
        Atom(element="C", x=4, y=0, z=0),
        Atom(element="H", x=5.09, y=0, z=0),
    ])
    ga = StandardGA(
        seed, logging.getLogger("test"), coordinates=InternalCoordinatesList(),
        pop_size=1, generations=0,
    )
    ga.setup()
    assert ga._overlapping_atoms(seed) is None
    touching = Molecule.from_molecule(seed)
    touching.atoms[3].x = touching.atoms[1].x + 0.982
    touching.atoms[3].y = 0
    touching.atoms[3].z = 0
    overlap = ga._overlapping_atoms(touching)
    assert overlap is not None
    assert "H2 and H4" in overlap
    assert "0.982" in overlap
    separated = Molecule.from_molecule(seed)
    separated.atoms[3].x = separated.atoms[1].x + 1.72
    assert ga._overlapping_atoms(separated) is None


def test_high_force_field_gradient_is_repeated_twice_then_discarded(molecule, monkeypatch):
    calls = {"minimize": 0}

    class FakeField:
        def Initialize(self):
            return None

        def Minimize(self, maxIts, forceTol, energyTol):
            calls["minimize"] += 1
            assert forceTol == 1e-4
            return 0

        def CalcEnergy(self):
            return 50.0

        def CalcGrad(self):
            return (100.0, 0.0, 0.0) * len(molecule.atoms)

    monkeypatch.setattr("rdkit.Chem.AllChem.UFFHasAllMoleculeParams", lambda rdkit_mol: True)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFGetMoleculeForceField", lambda rdkit_mol: FakeField())
    evaluator = RDKitEvaluator(forcefield="uff")
    evaluator.bind_connectivity(molecule)
    result = evaluator.optimize([molecule], max_iters=20)
    assert calls["minimize"] == 3
    assert result[0].properties["force_field_discarded"] is True
    assert result[0].properties["optimization_converged"] is False
    assert result[0].properties["force_field_gradient"] > 1e-4


def _scripted_force_field(molecule, energies, gradients, calls, tols=None, status=None):
    class FakeField:
        def __init__(self):
            self.step = 0

        def Initialize(self):
            calls["initialize"] = calls.get("initialize", 0) + 1
            return None

        def Minimize(self, maxIts, forceTol, energyTol):
            calls["minimize"] += 1
            if tols is not None:
                tols.append(forceTol)
            self.step += 1
            if self.step > len(energies):
                raise AssertionError("minimization continued after the scripted steps")
            return status

        def CalcEnergy(self):
            return energies[self.step - 1]

        def CalcGrad(self):
            value = gradients[self.step - 1]
            return (value, 0.0, 0.0) * len(molecule.atoms)

    return FakeField


def test_initial_relaxation_runs_ten_times_before_rejecting_a_high_gradient(molecule, monkeypatch):
    calls = {"minimize": 0}
    energies = [100.0, 90.0, 95.0] + [94.0] * 7
    gradients = [50.0, 40.0, 55.0] + [50.0] * 7
    field = _scripted_force_field(molecule, energies, gradients, calls)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFHasAllMoleculeParams", lambda rdkit_mol: True)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFGetMoleculeForceField", lambda rdkit_mol: field())
    evaluator = RDKitEvaluator(forcefield="uff")
    evaluator.bind_connectivity(molecule)
    result = evaluator.optimize(
        [molecule], max_iters=20, attempts=10, require_improvement=True,
    )[0]
    lines = result.properties["force_field_progress"].splitlines()
    assert calls["minimize"] == 10
    assert result.properties["energy"] == 94.0
    assert result.properties["force_field_discarded"] is True
    assert result.properties["force_field_attempts"] == 10
    assert result.properties["optimization_converged"] is False
    assert len(lines) == 10
    assert lines[0].startswith("Initial relaxation 1/10: energy 100 kcal/mol, gradient norm ")
    assert lines[2].startswith("Initial relaxation 3/10: energy 95 kcal/mol, gradient norm ")
    assert lines[9].startswith("Initial relaxation 10/10: energy 94 kcal/mol, gradient norm ")


def test_initial_relaxation_repeats_ten_times_while_both_decrease(molecule, monkeypatch):
    calls = {"minimize": 0}
    field = _scripted_force_field(
        molecule,
        [float(100 - step) for step in range(10)],
        [float(50 - step) for step in range(10)],
        calls,
    )
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFHasAllMoleculeParams", lambda rdkit_mol: True)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFGetMoleculeForceField", lambda rdkit_mol: field())
    evaluator = RDKitEvaluator(forcefield="uff")
    evaluator.bind_connectivity(molecule)
    result = evaluator.optimize(
        [molecule], max_iters=20, attempts=10, require_improvement=True,
    )[0]
    assert calls["minimize"] == 10
    assert result.properties["energy"] == 91.0
    assert result.properties["force_field_discarded"] is True
    assert result.properties["force_field_attempts"] == 10
    assert "energy 100 kcal/mol, gradient norm " in result.properties["force_field_progress"]
    assert "energy 91 kcal/mol, gradient norm " in result.properties["force_field_progress"]


def test_stalled_initial_step_retries_with_a_tighter_force_tolerance(molecule, monkeypatch):
    calls = {"minimize": 0}
    tols = []
    field = _scripted_force_field(
        molecule, [100.0, 100.0, 70.0], [50.0, 50.0, 0.0], calls, tols=tols,
    )
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFHasAllMoleculeParams", lambda rdkit_mol: True)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFGetMoleculeForceField", lambda rdkit_mol: field())
    evaluator = RDKitEvaluator(forcefield="uff")
    evaluator.bind_connectivity(molecule)
    result = evaluator.optimize(
        [molecule], max_iters=20, attempts=10, require_improvement=True,
    )[0]
    assert calls["minimize"] == 3
    assert tols == [1e-4, 1e-4, 1e-4]
    assert result.properties["energy"] == 70.0
    assert result.properties.get("force_field_discarded") is not True
    assert "energy 100 kcal/mol, gradient norm " in result.properties["force_field_progress"]
    assert "energy 70 kcal/mol, gradient norm " in result.properties["force_field_progress"]


def test_rdkit_convergence_tightens_the_next_initial_attempt(molecule, monkeypatch):
    calls = {"minimize": 0}
    tols = []
    field = _scripted_force_field(
        molecule, [100.0, 90.0, 80.0], [50.0, 40.0, 0.0], calls, tols=tols, status=0,
    )
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFHasAllMoleculeParams", lambda rdkit_mol: True)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFGetMoleculeForceField", lambda rdkit_mol: field())
    evaluator = RDKitEvaluator(forcefield="uff")
    evaluator.bind_connectivity(molecule)
    result = evaluator.optimize(
        [molecule], max_iters=20, attempts=10, require_improvement=True,
    )[0]
    assert tols == pytest.approx([1e-4, 1e-5, 1e-6])
    assert result.properties["energy"] == 80.0
    assert result.properties.get("force_field_discarded") is not True


def test_initial_relaxation_stops_once_the_gradient_is_small(molecule, monkeypatch):
    calls = {"minimize": 0}
    field = _scripted_force_field(molecule, [100.0, 80.0, 60.0, 40.0], [50.0, 30.0, 10.0, 0.0], calls)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFHasAllMoleculeParams", lambda rdkit_mol: True)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFGetMoleculeForceField", lambda rdkit_mol: field())
    evaluator = RDKitEvaluator(forcefield="uff")
    evaluator.bind_connectivity(molecule)
    result = evaluator.optimize(
        [molecule], max_iters=20, attempts=10, require_improvement=True,
    )[0]
    assert calls["minimize"] == 4
    assert result.properties["energy"] == 40.0
    assert result.properties.get("force_field_discarded") is not True
    assert result.properties["force_field_attempts"] == 4
    assert result.properties["optimization_converged"] is True


def test_initial_relaxation_requests_ten_improving_attempts(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class AwareEvaluator(Evaluator):
        def optimize(self, molecules, *, max_iters, attempts=3, require_improvement=False):
            self.calls.append((
                "optimize", len(molecules), max_iters, attempts, require_improvement,
            ))
            return super().optimize(molecules, max_iters=max_iters)

    evaluator = AwareEvaluator()
    ga = StandardGA(
        molecule, logging.getLogger("test"), evaluator=evaluator, coordinates=coordinates,
        pop_size=1, generations=0,
    )
    ga.run()
    assert evaluator.calls[0] == ("optimize", 1, 500, 10, True)


def test_initial_relaxation_logs_each_energy_and_gradient_norm(molecule, coordinates):
    runner = RecordingRunner()

    class LoggingEvaluator(Evaluator):
        def optimize(self, molecules, *, max_iters, attempts=3, require_improvement=False):
            result = Molecule.from_molecule(molecules[0])
            result.properties.update(
                energy=12.5,
                energy_unit="kcal/mol",
                force_field_discarded=True,
                force_field_gradient=0.000839301,
                force_field_attempts=attempts,
                force_field_progress=(
                    "Initial relaxation 1/10: energy 20 kcal/mol, gradient norm 1.5\n"
                    "Initial relaxation 10/10: energy 12.5 kcal/mol, gradient norm 0.4"
                ),
            )
            return [result]

    ga = StandardGA(
        molecule, runner, evaluator=LoggingEvaluator(), coordinates=coordinates,
        pop_size=1, generations=0,
    )
    with pytest.raises(ValueError, match="high force-field gradient"):
        ga.run()
    assert "Initial relaxation 1/10: energy 20 kcal/mol, gradient norm 1.5" in runner.log_string
    assert "Initial relaxation 10/10: energy 12.5 kcal/mol, gradient norm 0.4" in runner.log_string
    assert "after 10 optimizations" in runner.log_string


def test_low_force_field_gradient_stops_without_a_repeat(molecule, monkeypatch):
    calls = {"minimize": 0}

    class FakeField:
        def Initialize(self):
            return None

        def Minimize(self, maxIts, forceTol, energyTol):
            calls["minimize"] += 1
            return 0

        def CalcEnergy(self):
            return 50.0

        def CalcGrad(self):
            return (0.0,) * (len(molecule.atoms) * 3)

    monkeypatch.setattr("rdkit.Chem.AllChem.UFFHasAllMoleculeParams", lambda rdkit_mol: True)
    monkeypatch.setattr("rdkit.Chem.AllChem.UFFGetMoleculeForceField", lambda rdkit_mol: FakeField())
    evaluator = RDKitEvaluator(forcefield="uff")
    evaluator.bind_connectivity(molecule)
    result = evaluator.optimize([molecule], max_iters=20)
    assert calls["minimize"] == 1
    assert result[0].properties.get("force_field_discarded") is not True
    assert result[0].properties["optimization_converged"] is True


def test_nh2_hydrogen_swap_has_no_dihedral_rmsd():
    from molecular_qm_search.optimization.ga_population import dihedral_rmsd

    amine = Molecule(atoms=[
        Atom(element="C", x=0.0, y=1.2, z=0.0),
        Atom(element="C", x=0.0, y=0.0, z=0.0),
        Atom(element="N", x=1.40, y=0.0, z=0.0),
        Atom(element="H", x=1.85, y=0.90, z=0.0),
        Atom(element="H", x=1.85, y=-0.90, z=0.0),
    ])
    torsion = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180, 180)
    torsion.compute(amine)
    ga = StandardGA(
        amine, logging.getLogger("test"), coordinates=InternalCoordinatesList(elements=[torsion]),
        pop_size=2, generations=0,
    )
    ga.setup()
    shift = ga._dihedral_symmetries[0]
    assert shift == 180.0
    swapped = copy.deepcopy(ga.template_coords)
    angle = float(swapped.elements[0].get_actual_value(swapped.elements[0].value)) + shift
    low, high = swapped.elements[0].min_values[0], swapped.elements[0].max_values[0]
    swapped.elements[0].value = (angle - low) / (high - low)
    swapped.elements[0].real_values = [angle]
    assert ga.dihedral_rmsd(ga.template_coords, swapped) == pytest.approx(0.0, abs=1e-6)
    assert ga.dihedral_rmsd(swapped, ga.template_coords) == pytest.approx(0.0, abs=1e-6)
    assert dihedral_rmsd(ga.template_coords, swapped) == pytest.approx(abs(shift), abs=0.1)
    kept, prune = ga._unique_conformers([
        (1.0, ga.template_coords, "initial", 0),
        (2.0, swapped, "mutation-single", 1),
    ])
    assert len(kept) == 1
    assert prune["dropped"] == 1


def _search(molecule, torsion):
    ga = StandardGA(
        molecule, logging.getLogger("test"),
        coordinates=InternalCoordinatesList(elements=[torsion]),
        pop_size=2, generations=0,
    )
    ga.setup()
    return ga


def _offset(coords, degrees):
    moved = copy.deepcopy(coords)
    angle = float(moved.elements[0].get_actual_value(moved.elements[0].value)) + degrees
    low, high = moved.elements[0].min_values[0], moved.elements[0].max_values[0]
    moved.elements[0].value = (angle - low) / (high - low)
    moved.elements[0].real_values = [angle]
    return moved


def test_methyl_period_is_120_and_methylene_stays_distinct():
    methyl = Molecule(atoms=[
        Atom(element="C", x=0.0, y=1.2, z=0.0),
        Atom(element="C", x=0.0, y=0.0, z=0.0),
        Atom(element="C", x=1.50, y=0.0, z=0.0),
        Atom(element="H", x=1.86, y=1.03, z=0.0),
        Atom(element="H", x=1.86, y=-0.51, z=0.89),
        Atom(element="H", x=1.86, y=-0.51, z=-0.89),
    ])
    torsion = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180, 180)
    torsion.compute(methyl)
    ga = _search(methyl, torsion)
    assert ga._dihedral_symmetries == [120.0]
    turned = _offset(ga.template_coords, 120.0)
    assert ga.dihedral_rmsd(ga.template_coords, turned) == pytest.approx(0.0, abs=1e-6)
    turned = _offset(ga.template_coords, 240.0)
    assert ga.dihedral_rmsd(ga.template_coords, turned) == pytest.approx(0.0, abs=1e-6)
    turned = _offset(ga.template_coords, 60.0)
    assert ga.dihedral_rmsd(ga.template_coords, turned) == pytest.approx(60.0, abs=1e-6)

    methylene = Molecule(atoms=[
        Atom(element="C", x=0.0, y=1.2, z=0.0),
        Atom(element="C", x=0.0, y=0.0, z=0.0),
        Atom(element="C", x=1.50, y=0.0, z=0.0),
        Atom(element="H", x=1.90, y=0.94, z=0.30),
        Atom(element="H", x=1.90, y=-0.70, z=0.70),
        Atom(element="F", x=1.90, y=-0.30, z=-1.20),
    ])
    torsion = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180, 180)
    torsion.compute(methylene)
    ga = _search(methylene, torsion)
    assert ga._dihedral_symmetries == [None]
    turned = _offset(ga.template_coords, 180.0)
    assert ga.dihedral_rmsd(ga.template_coords, turned) == pytest.approx(180.0, abs=1e-6)


def test_four_identical_substituents_have_no_period():
    crowded = Molecule(atoms=[
        Atom(element="C", x=0.0, y=1.2, z=0.0),
        Atom(element="C", x=0.0, y=0.0, z=0.0),
        Atom(element="C", x=1.50, y=0.0, z=0.0),
        Atom(element="H", x=2.20, y=0.80, z=0.40),
        Atom(element="H", x=2.20, y=-0.80, z=0.40),
        Atom(element="H", x=2.20, y=0.0, z=-0.90),
        Atom(element="H", x=2.55, y=0.0, z=0.20),
    ])
    torsion = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180, 180)
    torsion.compute(crowded)
    with pytest.raises(ValueError, match="identical H substituents"):
        _search(crowded, torsion)
    amine = Molecule(atoms=[
        Atom(element="C", x=0.0, y=0.0, z=0.0),
        Atom(element="N", x=1.45, y=0.0, z=0.0),
        Atom(element="H", x=1.85, y=0.90, z=0.2),
        Atom(element="H", x=1.85, y=-0.90, z=0.2),
    ])

    class CrushHydrogens:
        def score(self, molecules):
            return self.optimize(molecules, max_iters=1)

        def optimize(self, molecules, *, max_iters):
            results = []
            for source in molecules:
                result = Molecule.from_molecule(source)
                result.atoms[3].x = result.atoms[2].x
                result.atoms[3].y = result.atoms[2].y
                result.atoms[3].z = result.atoms[2].z + 0.58
                result.properties["energy"] = 136.0
                result.properties["energy_unit"] = "kcal/mol"
                results.append(result)
            return results

    runner = RecordingRunner()
    ga = MinimizingGA(
        amine, runner, evaluator=CrushHydrogens(), coordinates=InternalCoordinatesList(),
        pop_size=1, generations=0,
    )
    ga.setup()
    assert ga._overlapping_atoms(amine) is None
    scored = ga.evaluate_population([InternalCoordinatesList()])
    assert scored == [None]
    assert "H3 and H4" in runner.log_string
    assert "0.580" in runner.log_string


def test_minimizing_ga_drops_a_conformer_whose_gradient_stays_high(molecule, coordinates):
    class DiscardEvaluator:
        def score(self, molecules):
            results = []
            for source in molecules:
                result = Molecule.from_molecule(source)
                result.properties["energy"] = 1.0
                result.properties["energy_unit"] = "kcal/mol"
                results.append(result)
            return results

        def optimize(self, molecules, *, max_iters):
            results = []
            for index, source in enumerate(molecules):
                result = Molecule.from_molecule(source)
                result.properties["energy"] = 1.0 + index
                result.properties["energy_unit"] = "kcal/mol"
                if index == 0:
                    result.properties["force_field_discarded"] = True
                    result.properties["force_field_gradient"] = 12.5
                results.append(result)
            return results

    runner = RecordingRunner()
    ga = MinimizingGA(
        molecule, runner, evaluator=DiscardEvaluator(), coordinates=coordinates,
        pop_size=2, generations=0,
    )
    ga.setup()
    scored = ga.evaluate_population([
        copy.deepcopy(ga.template_coords),
        copy.deepcopy(ga.template_coords),
    ])
    assert scored[0] is None
    assert scored[1][0] == 2.0
    assert "gradient stayed high (12.5)" in runner.log_string
    assert "two repeated optimizations" in runner.log_string


@pytest.mark.parametrize("ga_class", [StandardGA, MinimizingGA, DiversityGA])
def test_real_rdkit_ga_smoke(ga_class, tmp_path, monkeypatch):
    from molecular_qm_util import smiles_to_molecule, evaluate_molecules_rdkit

    monkeypatch.chdir(tmp_path)
    ga = ga_class(smiles_to_molecule("CCCC"), logging.getLogger("test"),
                  pop_size=3, num_confs=2, generations=0, parallel_children=1, max_iters=100,
                  mutation_rate=1.0)
    result = list(ga.run())
    assert result
    rescored = evaluate_molecules_rdkit(result, threads=1)
    for final, checked in zip(result, rescored):
        # Stored energy belongs to the returned geometry. OpenBabel SDF
        # conversion rounds positions to four decimals on the second evaluation.
        assert final.properties["energy"] == pytest.approx(checked.properties["energy"], abs=1e-4)
