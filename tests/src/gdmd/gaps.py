"""Gap-field construction utilities for gDMD."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def second_moment_gaps(
    x_dist: np.ndarray,
    y_dist: np.ndarray,
    *,
    weights: str = "inv_r",
    h: float | np.ndarray | None = None,
    eps: float = 1e-12,
    return_eigs: bool = False,
):
    """Compute axis-aligned second-moment gap proxies from neighbour offsets.

    The method forms, for each seed particle, a local second-moment tensor of
    neighbour offsets and returns ``sqrt(M11)`` and ``sqrt(M22)`` as the local
    directional gap proxies.
    """

    x = np.asarray(x_dist, dtype=float)
    y = np.asarray(y_dist, dtype=float)
    r = np.hypot(x, y) + eps

    if weights == "uniform":
        w = np.ones_like(r)
    elif weights == "inv_r":
        w = 1.0 / r
    elif weights == "gaussian":
        if h is None:
            hi = np.median(r, axis=1) + eps
        else:
            hi = np.asarray(h, dtype=float)
            if hi.ndim == 0:
                hi = np.full(r.shape[0], float(hi))
            hi = hi + eps
        w = np.exp(-(r / hi[:, None]) ** 2)
    else:
        raise ValueError("weights must be 'uniform', 'inv_r', or 'gaussian'.")

    wsum = np.sum(w, axis=1) + eps
    M11 = np.sum(w * x * x, axis=1) / wsum
    M22 = np.sum(w * y * y, axis=1) / wsum
    M12 = np.sum(w * x * y, axis=1) / wsum

    densx = np.sqrt(np.maximum(M11, 0.0))
    densy = np.sqrt(np.maximum(M22, 0.0))

    if not return_eigs:
        return densx, densy

    tr = M11 + M22
    det = M11 * M22 - M12 * M12
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)

    vx = M12
    vy = lam1 - M11
    norm = np.hypot(vx, vy) + eps
    e1 = np.stack((vx / norm, vy / norm), axis=1)
    return densx, densy, lam1, lam2, e1


def _same_neighbour_offsets(points: np.ndarray, neighbour_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Offsets from each point to a fixed set of neighbour indices."""

    x_dist = points[neighbour_indices, 0] - points[:, 0][:, None]
    y_dist = points[neighbour_indices, 1] - points[:, 1][:, None]
    return x_dist, y_dist


def compute_gap_snapshot_pair(
    seeds1: np.ndarray,
    seeds2: np.ndarray,
    xgrid: np.ndarray,
    *,
    n_neighbors: int = 15,
    gap_weights: str = "inv_r",
) -> tuple[np.ndarray, np.ndarray]:
    """Map particle gap proxies from one PIV image pair to an Eulerian grid.

    The same seed-neighbour connectivity, obtained from ``seeds1``, is used for
    ``seeds2``. This is required when forming the velocity-associated gDMD mode
    as the difference between exact and projected gap modes.
    """

    seeds1 = np.asarray(seeds1, dtype=float)
    seeds2 = np.asarray(seeds2, dtype=float)
    xgrid = np.asarray(xgrid, dtype=float)

    if seeds1.shape != seeds2.shape:
        raise ValueError("seeds1 and seeds2 must have the same shape.")
    if seeds1.ndim != 2 or seeds1.shape[1] != 2:
        raise ValueError("seeds1 and seeds2 must have shape (n_seeds, 2).")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2.")
    if n_neighbors >= seeds1.shape[0]:
        raise ValueError("n_neighbors must be smaller than the number of seeds.")

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(seeds1)
    _, neighbour_indices = knn.kneighbors(seeds1)
    neighbour_indices = neighbour_indices[:, 1:]

    x_dist1, y_dist1 = _same_neighbour_offsets(seeds1, neighbour_indices)
    densx1, densy1 = second_moment_gaps(x_dist1, y_dist1, weights=gap_weights)

    _, grid_neighbours1 = knn.kneighbors(xgrid)
    gap1 = np.empty((xgrid.shape[0], 2), dtype=float)
    gap1[:, 0] = np.mean(densx1[grid_neighbours1], axis=1)
    gap1[:, 1] = np.mean(densy1[grid_neighbours1], axis=1)

    x_dist2, y_dist2 = _same_neighbour_offsets(seeds2, neighbour_indices)
    densx2, densy2 = second_moment_gaps(x_dist2, y_dist2, weights=gap_weights)

    knn.fit(seeds2)
    _, grid_neighbours2 = knn.kneighbors(xgrid)
    gap2 = np.empty((xgrid.shape[0], 2), dtype=float)
    gap2[:, 0] = np.mean(densx2[grid_neighbours2], axis=1)
    gap2[:, 1] = np.mean(densy2[grid_neighbours2], axis=1)

    return gap1, gap2
