"""Neighbor edges of a multiplicatively weighted Voronoi diagram on the dihedral torus.

The distance is the circular dihedral RMS used by ``dihedral_rmsd``. Site ``i``
owns a witness ``x`` when ``distance(x, site_i) / weight_i`` is smallest. An
edge is recorded when a witness finds two sites within ``neighbor_gap`` of each
other and strictly closer than every other site. Cell vertices are not built.
"""
from __future__ import annotations

import numpy as np


def circular_dihedral_differences(left, right, symmetries=None):
    """Per-angle circular separation in degrees, matching ``dihedral_rmsd``.

    ``left`` and ``right`` broadcast over their leading axes. The last axis is
    the dihedral. ``symmetries`` has one entry per dihedral: ``None`` for an
    ordinary torsion, or the degree shift between indistinguishable terminal
    hydrogens.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape[-1] != right.shape[-1]:
        raise ValueError(
            f"Dihedral lists differ in length ({left.shape[-1]} and {right.shape[-1]})"
        )
    width = left.shape[-1]
    if symmetries is not None and len(symmetries) != width:
        raise ValueError(
            f"Dihedral symmetries ({len(symmetries)}) do not match the {width} dihedrals"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Dihedral angles must be finite")
    difference = np.abs(left - right) % 360.0
    difference = np.where(difference > 180.0, 360.0 - difference, difference)
    if symmetries is None:
        return difference
    for index, shift in enumerate(symmetries):
        if shift is None:
            continue
        shift = float(shift)
        if not np.isfinite(shift):
            raise ValueError(f"Dihedral {index} has a non-finite symmetry shift")
        for sign in (1.0, -1.0):
            shifted = np.abs(left[..., index] - (right[..., index] - sign * shift)) % 360.0
            shifted = np.where(shifted > 180.0, 360.0 - shifted, shifted)
            difference[..., index] = np.minimum(difference[..., index], shifted)
    return difference


def circular_dihedral_rmsd(left, right, symmetries=None) -> float:
    """Circular root-mean-square dihedral difference in degrees."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError(
            f"Dihedral RMSD expects one angle vector per conformer, "
            f"got shapes {left.shape} and {right.shape}"
        )
    if left.shape != right.shape:
        raise ValueError(
            f"Dihedral lists differ in length ({left.shape[0]} and {right.shape[0]})"
        )
    if left.size == 0:
        return 0.0
    difference = circular_dihedral_differences(left, right, symmetries)
    return float(np.sqrt(np.mean(difference * difference)))


def weighted_voronoi_neighbor_edges(
    points,
    weights,
    *,
    bounds,
    n_witnesses,
    neighbor_gap,
    seed,
    symmetries=None,
) -> list[tuple[int, int]]:
    """Undirected neighbor edges on the periodic dihedral torus.

    ``points`` has shape ``(sites, dihedrals)`` and holds angles in degrees.
    ``weights`` is one positive weight per site. ``bounds`` has shape
    ``(dihedrals, 2)`` and is the box the witnesses are drawn from. A larger
    weight pushes the boundary farther from that site: a ratio of 2:1 places it
    twice as far from the heavier site as from the lighter one.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"Dihedral points must have shape (sites, dihedrals), got {points.shape}")
    if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
        raise ValueError(
            f"Expected one weight per site ({points.shape[0]}), got shape {weights.shape}"
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f"Voronoi weights must be positive and finite, got {weights}")
    if isinstance(n_witnesses, bool) or not isinstance(n_witnesses, int) or n_witnesses < 1:
        raise ValueError(f"n_witnesses must be a positive integer, got {n_witnesses!r}")
    if not np.isfinite(neighbor_gap) or neighbor_gap <= 0:
        raise ValueError(f"neighbor_gap must be a positive finite number, got {neighbor_gap!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Witness seed must be an integer, got {seed!r}")
    n_sites, width = points.shape
    if bounds.shape != (width, 2):
        raise ValueError(
            f"Dihedral bounds must have shape ({width}, 2), got {bounds.shape}"
        )
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"Each dihedral bound must be a finite interval, got {bounds}")
    if n_sites < 2:
        return []

    low = bounds[:, 0]
    span = bounds[:, 1] - bounds[:, 0]
    witnesses = low + np.random.default_rng(seed).random((n_witnesses, width)) * span
    difference = circular_dihedral_differences(
        witnesses[:, None, :], points[None, :, :], symmetries,
    )
    if width == 0:
        distance = np.zeros((n_witnesses, n_sites), dtype=float)
    else:
        distance = np.sqrt(np.mean(difference * difference, axis=-1))
    weighted = distance / weights
    order = np.argsort(weighted, axis=1)
    rows = np.arange(n_witnesses)
    first = weighted[rows, order[:, 0]]
    second = weighted[rows, order[:, 1]]
    close = second <= first * (1.0 + neighbor_gap)
    if n_sites > 2:
        third = weighted[rows, order[:, 2]]
        close = close & (first < third)
    edges = {
        (int(left), int(right)) if left < right else (int(right), int(left))
        for left, right in zip(order[close, 0], order[close, 1])
    }
    return sorted(edges)
