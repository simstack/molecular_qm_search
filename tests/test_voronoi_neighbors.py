import logging

import numpy as np
import pytest

from molecular_qm_models import Atom, InternalCoordinatesList, InternalDihedralCoordinate, Molecule
from molecular_qm_search.optimization.ga_population import PopulationGenerator, dihedral_rmsd
from molecular_qm_search.optimization.voronoi_method import (
    VoronoiNeighborSearch,
    generate_voronoi_conformers,
)
from molecular_qm_search.optimization.voronoi_neighbors import (
    circular_dihedral_rmsd,
    weighted_voronoi_neighbor_edges,
)


def _coordinates(angles, low=-180.0, high=180.0):
    span = high - low
    elements = []
    for index, angle in enumerate(angles):
        coord = InternalDihedralCoordinate.initialize(0, 1, 2, 3 + index, low, high)
        coord.value = (float(angle) - low) / span
        elements.append(coord)
    return InternalCoordinatesList(elements=elements)


def _edges(points, weights, *, seed=1, n_witnesses=4000, neighbor_gap=0.05, low=0.0, high=360.0):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    bounds = np.tile([low, high], (points.shape[1], 1))
    return weighted_voronoi_neighbor_edges(
        points,
        np.asarray(weights, dtype=float),
        bounds=bounds,
        n_witnesses=n_witnesses,
        neighbor_gap=neighbor_gap,
        seed=seed,
    )


def test_two_sites_share_one_edge_for_equal_and_ratio_weights():
    points = [0.0, 180.0]
    assert _edges(points, [1.0, 1.0]) == [(0, 1)]
    assert _edges(points, [2.0, 1.0]) == [(0, 1)]


def test_periodic_pair_across_the_angle_cut_is_a_neighbor():
    edges = _edges([1.0, 180.0, 359.0], [1.0, 1.0, 1.0])
    assert (0, 2) in edges


def test_tiny_weight_between_two_heavy_sites_is_isolated():
    edges = _edges([0.0, 10.0, 180.0], [2.0, 1.0e-6, 2.0], n_witnesses=8000, neighbor_gap=0.05)
    assert edges == [(0, 2)]


def test_angle_pairs_match_dihedral_rmsd():
    cases = [
        ([10.0, -20.0], [350.0, 160.0], None),
        ([0.0, 90.0], [180.0, -90.0], None),
        ([0.0], [359.0], None),
        ([0.0], [120.0], [120.0]),
        ([0.0], [240.0], [120.0]),
        ([0.0], [60.0], [120.0]),
        ([0.0], [180.0], [180.0]),
        ([0.0], [90.0], [180.0]),
        ([15.0, 40.0], [15.0 + 120.0, 40.0], [120.0, None]),
    ]
    for left, right, symmetries in cases:
        measured = circular_dihedral_rmsd(left, right, symmetries)
        reference = dihedral_rmsd(_coordinates(left), _coordinates(right), symmetries)
        assert measured == pytest.approx(reference)
    assert circular_dihedral_rmsd([0.0], [359.0]) == pytest.approx(1.0)
    assert circular_dihedral_rmsd([0.0], [120.0], [120.0]) == pytest.approx(0.0)
    assert circular_dihedral_rmsd([0.0], [240.0], [120.0]) == pytest.approx(0.0)
    assert circular_dihedral_rmsd([0.0], [60.0], [120.0]) == pytest.approx(60.0)
    assert circular_dihedral_rmsd([0.0], [180.0], [180.0]) == pytest.approx(0.0)
    assert circular_dihedral_rmsd([0.0], [90.0], [180.0]) == pytest.approx(90.0)
    assert circular_dihedral_rmsd([0.0], [180.0]) == pytest.approx(180.0)


def test_invalid_weights_gap_and_witness_count_raise():
    points = np.array([[0.0], [90.0]])
    bounds = np.array([[0.0, 360.0]])
    with pytest.raises(ValueError, match="positive and finite"):
        weighted_voronoi_neighbor_edges(
            points, [1.0, 0.0], bounds=bounds, n_witnesses=10, neighbor_gap=0.1, seed=1,
        )
    with pytest.raises(ValueError, match="neighbor_gap"):
        weighted_voronoi_neighbor_edges(
            points, [1.0, 1.0], bounds=bounds, n_witnesses=10, neighbor_gap=0.0, seed=1,
        )
    with pytest.raises(ValueError, match="n_witnesses"):
        weighted_voronoi_neighbor_edges(
            points, [1.0, 1.0], bounds=bounds, n_witnesses=0, neighbor_gap=0.1, seed=1,
        )


def test_crossover_uses_neighbor_edges_and_rejects_an_empty_graph():
    molecule = Molecule(atoms=[
        Atom(element="C", x=0, y=1, z=0),
        Atom(element="C", x=0, y=0, z=0),
        Atom(element="C", x=1.5, y=0, z=0),
        Atom(element="C", x=1.5, y=1, z=1),
    ])
    coordinates = _coordinates([0.0])
    generator = PopulationGenerator(
        molecule, coordinates=coordinates, seed=1, mutation_rate=0.0, dihedral_interval=30.0,
    )
    parents = []
    history = []
    for index, angle in enumerate((0.0, 90.0, 180.0)):
        coords = _coordinates([angle])
        history.append({"op": "I", "index": index})
        parents.append((coords, "initial", float(index), index))
    with pytest.raises(ValueError, match="neighbor edge"):
        generator.reproduce(parents, len(parents) + 1, 20.0, history, neighbor_edges=[])
    offspring = generator.reproduce(
        parents, len(parents) + 1, 20.0, history, neighbor_edges=[(0, 2)],
    )
    operation = history[-1]
    assert operation["op"] == "X"
    assert operation["a1"] == 0
    assert operation["a2"] == 2
    assert offspring[-1][1] == "crossover"


class _Evaluator:
    def score(self, molecules):
        results = [Molecule.from_molecule(molecule) for molecule in molecules]
        for molecule in results:
            molecule.properties["energy"] = 1.0
        return results

    def optimize(self, molecules, *, max_iters):
        raise AssertionError("The neighbor search scores conformers and does not relax them")


def _butane():
    return Molecule(atoms=[
        Atom(element="C", x=0, y=1, z=0),
        Atom(element="C", x=0, y=0, z=0),
        Atom(element="C", x=1.5, y=0, z=0),
        Atom(element="C", x=1.5, y=1, z=1),
    ])


def test_search_publishes_the_neighbor_graph():
    runner = logging.getLogger("voronoi-neighbors")
    search = VoronoiNeighborSearch(
        _butane(),
        runner,
        evaluator=_Evaluator(),
        coordinates=_coordinates([0.0]),
        pop_size=4,
        generations=0,
        num_confs=4,
        mutation_rate=1.0,
        seed=1,
        prune_rms_thresh=20.0,
        weight_temperature=1.0,
        n_witnesses=128,
        neighbor_gap=0.05,
    )
    molecules = search.run()
    assert list(molecules)
    assert runner.neighbor_graph.name == "Weighted dihedral neighbor graph"
    assert runner.neighbor_graph.heading == ["site_a", "site_b"]
    assert search.template_coords.elements


def test_rigid_molecule_publishes_an_empty_neighbor_graph():
    runner = logging.getLogger("voronoi-rigid")
    molecules = generate_voronoi_conformers(
        _butane(),
        node_runner=runner,
        evaluator=_Evaluator(),
        coordinates=InternalCoordinatesList(),
        pop_size=1,
        generations=0,
        num_confs=1,
        weight_temperature=1.0,
        n_witnesses=8,
        neighbor_gap=0.05,
    )
    assert len(list(molecules)) == 1
    assert runner.neighbor_graph.row == []


def test_non_positive_weight_temperature_is_rejected():
    with pytest.raises(ValueError, match="weight_temperature"):
        VoronoiNeighborSearch(
            _butane(),
            logging.getLogger("voronoi-temperature"),
            evaluator=_Evaluator(),
            coordinates=_coordinates([0.0]),
            weight_temperature=0.0,
            n_witnesses=8,
            neighbor_gap=0.05,
        )
