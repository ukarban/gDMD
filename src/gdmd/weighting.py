"""Spatial weighting utilities for gDMD."""

from __future__ import annotations

import numpy as np


def tke_weights_from_velocity(
    u1dmd: np.ndarray,
    u2dmd: np.ndarray,
    n_points: int,
    *,
    alpha: float = 0.5,
    floor: float = 0.05,
    eps: float = 1e-14,
) -> np.ndarray:
    """Compute a normalized TKE-based spatial weight from velocity snapshots."""

    if not (0.0 <= floor <= 1.0):
        raise ValueError("floor must lie in [0, 1].")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    U = np.concatenate((u1dmd, u2dmd), axis=1).reshape(n_points, 2, -1)
    Up = U - np.mean(U, axis=2, keepdims=True)
    tke = 0.5 * np.mean(Up[:, 0, :] ** 2 + Up[:, 1, :] ** 2, axis=1)
    tke_norm = tke / (np.max(tke) + eps)
    return floor + (1.0 - floor) * tke_norm**alpha


def apply_spatial_weights(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``sqrt(W)`` to an interleaved two-component state matrix."""

    sqrtw = np.sqrt(np.repeat(np.asarray(w), 2))[:, None]
    return sqrtw * X, sqrtw


def undo_spatial_weights(Phi_w: np.ndarray, sqrtw: np.ndarray, *, eps: float = 1e-14) -> np.ndarray:
    """Map weighted modes back to unweighted coordinates."""

    return Phi_w / (sqrtw + eps)
