"""State-vector and DMD-window utilities shared by all gDMD examples."""

from __future__ import annotations

import numpy as np


def restrict_state_matrix(field: np.ndarray, point_mask: np.ndarray) -> np.ndarray:
    """Restrict a two-component field to point-major DMD matrix form.

    Parameters
    ----------
    field:
        Array with shape ``(n_points, 2, n_snapshots)``.
    point_mask:
        Boolean mask selecting the Eulerian points used in the DMD window.

    Returns
    -------
    X:
        Snapshot matrix with shape ``(2*n_selected_points, n_snapshots)`` and
        state ordering
        ``[q_x(1), q_y(1), q_x(2), q_y(2), ..., q_x(N), q_y(N)]``.
    """

    field = np.asarray(field)
    point_mask = np.asarray(point_mask, dtype=bool)
    if field.ndim != 3 or field.shape[1] != 2:
        raise ValueError("field must have shape (n_points, 2, n_snapshots).")
    if point_mask.ndim != 1 or point_mask.size != field.shape[0]:
        raise ValueError("point_mask must be a one-dimensional mask over field points.")

    restricted = field[point_mask, :, :]
    n_points = restricted.shape[0]
    n_snapshots = restricted.shape[2]
    return restricted.reshape(2 * n_points, n_snapshots)


def rectangular_window_mask(
    xgrid: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    centers: np.ndarray | None = None,
    radius: float | None = None,
    radius_buffer: float = 5.0e-4,
) -> np.ndarray:
    """Return a rectangular DMD-window mask with optional cylinder exclusion."""

    xgrid = np.asarray(xgrid, dtype=float)
    if xgrid.ndim != 2 or xgrid.shape[1] != 2:
        raise ValueError("xgrid must have shape (n_points, 2).")

    mask = (
        (xgrid[:, 0] > xlim[0])
        & (xgrid[:, 0] < xlim[1])
        & (xgrid[:, 1] > ylim[0])
        & (xgrid[:, 1] < ylim[1])
    )

    if centers is not None and radius is not None:
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        for cx, cy in centers:
            mask &= np.hypot(xgrid[:, 0] - cx, xgrid[:, 1] - cy) > radius + radius_buffer

    return mask
