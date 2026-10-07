import copy
import logging
import random

import pytest

from molecular_qm_models import Atom, Molecule, InternalCoordinatesList, InternalDihedralCoordinate
from molecular_qm_models.internal_coordinates import InternalCoordinateBondType
from molecular_qm_search.optimization.ga_population import PopulationGenerator
from molecular_qm_search.optimization.ga_method import StandardGA, MinimizingGA, DiversityGA
from molecular_qm_search.optimization.lib.ga_evaluation import validate_results


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
            molecule.atoms[0].y = 2.5
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
    assert [p.elements[0].value for p, _ in pop1] == [p.elements[0].value for p, _ in pop2]
    before = copy.deepcopy(pop1[1][0])
    result = first.molecule_from_coordinates(pop1[1][0])
    assert result is not molecule
    assert pop1[1][0].model_dump() == before.model_dump()
    assert molecule.model_dump() == original_molecule.model_dump()
    assert coordinates.model_dump() == original_coordinates.model_dump()
    assert random.getstate() == state


def test_double_bond_mutation_flips_relative_to_parent(molecule, coordinates):
    coordinates.elements[0].bond_type = InternalCoordinateBondType.DOUBLE
    generator = PopulationGenerator(molecule, coordinates=coordinates, mutation_rate=1,
                                    crossover_rate=0, dihedral_interval=0)
    parents = generator.generate(1)
    offspring = generator.reproduce(parents, 2)
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
                  coordinates=coordinates, pop_size=3, num_confs=2, generations=1)
    ga.setup()
    assert evaluator.calls == []
    result = ga.run()
    assert result
    assert any(call[0] == "optimize" for call in evaluator.calls)
    assert all(m.atoms[0].y == 2.5 for m in result)
    assert all(m.properties["energy"] == -2 for m in result)
    assert all(m.properties["method"] == "test" for m in result)
    if ga_class is not MinimizingGA:
        assert any(call[0] == "score" for call in evaluator.calls)


def test_rigid_molecule_uses_backend_without_rdkit(molecule, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = Evaluator()
    ga = StandardGA(molecule, logging.getLogger("test"), evaluator=evaluator,
                    coordinates=InternalCoordinatesList())
    assert len(ga.run()) == 1
    assert evaluator.calls == [("optimize", 1, 500)]


def test_smart_optimization_preserves_order_and_partial_results(molecule, coordinates):
    evaluator = Evaluator()
    ga = MinimizingGA(molecule, logging.getLogger("test"), evaluator=evaluator,
                      coordinates=coordinates, smart_opt=True, max_iters=100)
    ga.setup()
    decisions = iter([True, False, True])
    ga.smart_optimizer.should_discard = lambda *args: next(decisions)
    population = [ind for ind, _ in ga.population_generator.generate(3)]
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
        pop_size=12, num_confs=6, generations=2, mutation_rate=1.0, crossover_rate=0.5, seed=1,
    )
    conformers = ga.run()
    assert len(conformers) == 6
    assert "Population:     12" in runner.log_string
    assert "Dihedral Types:" in runner.log_string
    assert runner.log_string == "\n".join(runner.info_messages) + "\n"
    chart = runner.energy_chart
    assert [series.yKey for series in chart.series] == ["min-energy", "max-energy"]
    assert [series.title for series in chart.series] == ["min-energy", "max-energy"]
    assert [row["iteration"] for row in chart.data] == [0, 1, 2]
    assert all("min-energy" in row and "max-energy" in row for row in chart.data)
    table = runner.operator_stats
    assert table.heading == ["change", "produced", "elite_survivors", "success_rate"]
    changes = [row["change"] for row in table.row]
    assert "initial" in changes
    assert "crossover" in changes
    assert any(change.startswith("mutation-") for change in changes)
    for row in table.row:
        assert 0 <= row["success_rate"] <= 1
        assert row["elite_survivors"] <= row["produced"]
    histogram = runner.energy_histogram
    assert histogram.series[0].type == "column"
    assert histogram.series[0].yKey == "count"
    assert sum(row["count"] for row in histogram.data) == len(conformers)
    assert runner.diversity_chart.series[0].type == "scatter"
    assert len(runner.diversity_chart.data) == 12
    assert {key for row in runner.diversity_chart.data for key in row} == {"pc1", "pc2"}


def test_charts_refresh_on_every_iteration(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RecordingRunner()
    seen = []

    class WatchingEvaluator(DistinctGeometryEvaluator):
        def score(self, molecules):
            chart = getattr(runner, "energy_chart", None)
            if chart is not None:
                seen.append({
                    "iterations": [row["iteration"] for row in chart.data],
                    "histogram": sum(row["count"] for row in runner.energy_histogram.data),
                    "diversity": len(runner.diversity_chart.data),
                })
            return super().score(molecules)

    ga = StandardGA(
        molecule, runner, evaluator=WatchingEvaluator(), coordinates=coordinates,
        pop_size=4, num_confs=4, generations=2, mutation_rate=1.0, seed=1,
    )
    ga.run()
    assert seen == [
        {"iterations": [0], "histogram": 4, "diversity": 4},
        {"iterations": [0, 1], "histogram": 4, "diversity": 4},
        {"iterations": [0, 1, 2], "histogram": 4, "diversity": 4},
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


class SpreadEvaluator:
    def __init__(self, spans):
        self.spans = spans
        self.calls = 0

    def score(self, molecules):
        if self.calls >= len(self.spans):
            raise ValueError("SpreadEvaluator received more score calls than spans")
        span = self.spans[self.calls]
        self.calls += 1
        if len(molecules) < 2:
            raise ValueError("SpreadEvaluator needs at least two molecules")
        results = []
        for index, molecule in enumerate(molecules):
            result = Molecule.from_molecule(molecule)
            result.properties["energy"] = float(span * index / (len(molecules) - 1))
            results.append(result)
        return results

    def optimize(self, molecules, *, max_iters):
        results = [Molecule.from_molecule(molecule) for molecule in molecules]
        for index, result in enumerate(results):
            result.properties["energy"] = float(index)
        return results


def test_energy_plot_drops_iterations_until_range_settles(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RecordingRunner()
    ga = StandardGA(
        molecule, runner, evaluator=SpreadEvaluator([100.0, 80.0, 10.0, 8.0]),
        coordinates=coordinates, pop_size=4, num_confs=4, generations=3, seed=1,
    )
    ga.run()
    assert [row["iteration"] for row in runner.energy_chart.data] == [2, 3]
    assert "Dropped 2 initial iterations" in runner.log_string


def test_restart_restores_generator_random_state(molecule, coordinates, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ga = StandardGA(molecule, logging.getLogger("test"), evaluator=Evaluator(), coordinates=coordinates)
    ga.setup()
    population = ga.population_generator.generate(4)
    ga.save_state(population, 2)
    expected = ga.reproduce(population[:1], 4)
    restored, generation = ga.load_state()
    actual = ga.reproduce(restored[:1], 4)
    assert generation == 2
    assert [p.elements[0].value for p, _ in actual] == [p.elements[0].value for p, _ in expected]


@pytest.mark.parametrize("ga_class", [StandardGA, MinimizingGA, DiversityGA])
def test_real_rdkit_ga_smoke(ga_class, tmp_path, monkeypatch):
    from molecular_qm_util import smiles_to_molecule, score_molecules_rdkit

    monkeypatch.chdir(tmp_path)
    ga = ga_class(smiles_to_molecule("CCCC"), logging.getLogger("test"),
                  pop_size=3, num_confs=2, generations=0, parallel_children=1, max_iters=100)
    result = list(ga.run())
    assert result
    rescored = score_molecules_rdkit(result, threads=1)
    for final, checked in zip(result, rescored):
        # Stored energy belongs to the returned geometry. OpenBabel SDF
        # conversion rounds positions to four decimals on the second evaluation.
        assert final.properties["energy"] == pytest.approx(checked.properties["energy"], abs=1e-4)
