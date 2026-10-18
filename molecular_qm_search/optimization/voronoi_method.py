"""Conformer search that crosses dihedrals along weighted Voronoi neighbor edges."""
from __future__ import annotations

import numpy as np

from molecular_qm_models import Molecule
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

from molecular_qm_search.optimization.ga_method import StandardGA
from molecular_qm_search.optimization.voronoi_neighbors import weighted_voronoi_neighbor_edges


def assign_neighbor_graph(node_runner, edges) -> SimpleTable:
    """Store the latest generation's undirected neighbor edges on the node runner."""
    table = SimpleTable(name="Weighted dihedral neighbor graph")
    table.add_column("site_a", SimpleTableColumnType.NUMBER)
    table.add_column("site_b", SimpleTableColumnType.NUMBER)
    for left, right in edges:
        table.add_row({"site_a": int(left), "site_b": int(right)})
    node_runner.neighbor_graph = table
    return table


class VoronoiNeighborSearch(StandardGA):
    """Score conformers, then cross only pairs that share a weighted Voronoi face."""

    def __init__(self, initial_mol, node_runner, *, weight_temperature, n_witnesses, neighbor_gap, **kwargs):
        if not np.isfinite(weight_temperature) or weight_temperature <= 0:
            raise ValueError(
                f"weight_temperature must be a positive finite energy scale, got {weight_temperature!r}"
            )
        if isinstance(n_witnesses, bool) or not isinstance(n_witnesses, int) or n_witnesses < 1:
            raise ValueError(f"n_witnesses must be a positive integer, got {n_witnesses!r}")
        if not np.isfinite(neighbor_gap) or neighbor_gap <= 0:
            raise ValueError(f"neighbor_gap must be a positive finite number, got {neighbor_gap!r}")
        super().__init__(initial_mol, node_runner, **kwargs)
        self.weight_temperature = float(weight_temperature)
        self.n_witnesses = n_witnesses
        self.neighbor_gap = float(neighbor_gap)

    def reproduce(self, best_individuals, target_size):
        edges = self._neighbor_edges(best_individuals)
        assign_neighbor_graph(self.node_runner, edges)
        self._report(
            f"Weighted neighbor graph: {len(edges)} edges among {len(best_individuals)} conformers."
        )
        return self.population_generator.reproduce(
            best_individuals, target_size, self.prune_rms_thresh, self._history,
            neighbor_edges=edges,
        )

    def _neighbor_edges(self, parents) -> list[tuple[int, int]]:
        if self._dihedral_symmetries is None:
            raise ValueError("Dihedral symmetries are unavailable before setup")
        if not parents:
            raise ValueError("Cannot build a neighbor graph of an empty population")
        points = []
        energies = []
        for coords, _origin, energy, _history_index in parents:
            if energy is None or not np.isfinite(energy):
                raise ValueError("Neighbor weights require a finite conformer energy")
            points.append([
                float(coord.get_actual_value(coord.value)) for coord in coords.elements
            ])
            energies.append(float(energy))
        points = np.asarray(points, dtype=float)
        energies = np.asarray(energies, dtype=float)
        weights = np.exp(-(energies - float(np.min(energies))) / self.weight_temperature)
        bounds = np.asarray([
            [float(coord.min_values[0]), float(coord.max_values[0])]
            for coord in self.template_coords.elements
        ], dtype=float)
        return weighted_voronoi_neighbor_edges(
            points,
            weights,
            bounds=bounds,
            n_witnesses=self.n_witnesses,
            neighbor_gap=self.neighbor_gap,
            seed=self.seed,
            symmetries=self._dihedral_symmetries,
        )


def generate_voronoi_conformers(initial_mol: Molecule, **kwargs):
    search = VoronoiNeighborSearch(initial_mol, **kwargs)
    molecules = search.run()
    if getattr(search.node_runner, "neighbor_graph", None) is None:
        if search.template_coords.elements:
            raise ValueError("Voronoi search did not produce a neighbor graph")
        assign_neighbor_graph(search.node_runner, [])
    return molecules
