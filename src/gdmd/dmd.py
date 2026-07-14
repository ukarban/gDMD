"""Dynamic mode decomposition routines used by the cylinder gDMD example.

The main routine, :func:`dmd_pair`, implements the paired-snapshot formulation
used in Tu et al. (2014).  It returns both exact and projected DMD modes because
the gDMD velocity-associated modes are obtained from their difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy import linalg
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

SortMode = Literal["none", "abs_freq", "optimal", "amplitude"]


@dataclass(frozen=True)
class DMDResult:
    """Container for a paired-snapshot DMD calculation.

    Attributes
    ----------
    eigenvalues:
        Discrete-time DMD eigenvalues.
    modes_exact:
        Exact DMD modes, lying in the range of the second snapshot matrix.
    modes_projected:
        Projected/standard DMD modes, lying in the range of the first snapshot
        matrix used for the low-rank projection.
    amplitudes:
        Optional optimal amplitudes, returned only when amplitude-based sorting
        is requested.
    """

    eigenvalues: np.ndarray
    modes_exact: np.ndarray
    modes_projected: np.ndarray
    amplitudes: np.ndarray | None = None


def _check_inputs(X1: np.ndarray, X2: np.ndarray, rank: int, dt: float) -> None:
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must be two-dimensional snapshot matrices.")
    if X1.shape != X2.shape:
        raise ValueError(f"X1 and X2 must have the same shape, got {X1.shape} and {X2.shape}.")
    if rank < 1:
        raise ValueError("rank must be positive.")
    if rank > min(X1.shape):
        raise ValueError(f"rank={rank} exceeds min(X1.shape)={min(X1.shape)}.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")


def _truncated_svd(X: np.ndarray, rank: int, *, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the leading singular triplet of ``X``.

    Dense real matrices use randomized SVD for speed. Complex matrices use
    ``scipy.sparse.linalg.svds`` when possible and fall back to dense SVD for
    small or full-rank requests.
    """

    min_dim = min(X.shape)
    is_complex = np.iscomplexobj(X)

    if is_complex and rank < min_dim:
        u, s, vh = svds(X, k=rank, which="LM")
        return u[:, ::-1], s[::-1], vh[::-1, :]

    if is_complex:
        u, s, vh = linalg.svd(X, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank, :]

    return randomized_svd(X, n_components=rank, n_iter=30, random_state=random_state)


def _optimal_amplitudes(Phi: np.ndarray, mu: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute optimal DMD amplitudes for a fixed set of modes/eigenvalues."""

    n_modes = Phi.shape[1]
    n_snapshots = X.shape[1]
    vand = np.vander(mu, N=n_snapshots, increasing=True)

    gram_phi = Phi.conj().T @ Phi
    gram_vand = vand @ vand.conj().T
    lhs = gram_phi * gram_vand

    rhs = np.empty(n_modes, dtype=np.complex128)
    for j in range(n_modes):
        rhs[j] = (Phi[:, j].conj().T @ X @ vand[j, :].reshape(-1, 1)).item()

    try:
        return linalg.solve(lhs, rhs)
    except linalg.LinAlgError:
        return linalg.lstsq(lhs, rhs, rcond=None)[0]


def dmd_pair(
    X1: np.ndarray,
    X2: np.ndarray,
    rank: int,
    dt: float = 1.0,
    *,
    center: bool = True,
    sort_modes: SortMode | str = "none",
    return_amplitudes: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute paired-snapshot DMD modes.

    Parameters
    ----------
    X1, X2:
        Snapshot-pair matrices with identical shape ``(n_state, n_pairs)``.
        Each column pair is interpreted as ``X2[:, k] ≈ A X1[:, k]``.
    rank:
        Truncation rank.
    dt:
        Sampling delay associated with each snapshot pair. This is used only
        for frequency-based sorting.
    center:
        If ``True``, subtract the row-wise mean of each snapshot matrix before
        computing DMD.
    sort_modes:
        ``"none"`` sorts by eigenvalue magnitude, ``"abs_freq"`` sorts by
        increasing absolute frequency, and ``"optimal"``/``"amplitude"`` sorts
        by optimal DMD amplitude.
    return_amplitudes:
        If ``True``, return a fourth output containing optimal amplitudes when
        they are computed; otherwise it is ``None``.
    random_state:
        Seed used by randomized SVD for real dense matrices.

    Returns
    -------
    eigenvalues, modes_exact, modes_projected[, amplitudes]
        Discrete-time eigenvalues and the exact/projected DMD modes.
    """

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    _check_inputs(X1, X2, rank, dt)

    if center:
        X1n = X1 - np.mean(X1, axis=1, keepdims=True)
        X2n = X2 - np.mean(X2, axis=1, keepdims=True)
    else:
        X1n = X1.copy()
        X2n = X2.copy()

    U, singular_values, Vh = _truncated_svd(X1n, rank, random_state=random_state)
    V = Vh.conj().T
    inv_s = np.diag(1.0 / singular_values)

    Atilde = U.conj().T @ X2n @ V @ inv_s
    eigenvalues, W = linalg.eig(Atilde)

    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    W = W[:, order]

    safe_eigs = np.where(np.abs(eigenvalues) > 0.0, eigenvalues, np.finfo(float).eps)
    modes_exact = X2n @ V @ inv_s @ W @ np.diag(1.0 / safe_eigs)
    modes_projected = U @ W

    amplitudes = None
    sort_key = sort_modes.lower()
    if sort_key in {"optimal", "optimal_amplitude", "opt", "amplitude"}:
        Xfit = np.concatenate([X1n, X2n[:, -1:].copy()], axis=1)
        amplitudes = _optimal_amplitudes(modes_exact, eigenvalues, Xfit)
        order = np.argsort(np.abs(amplitudes))[::-1]
        eigenvalues = eigenvalues[order]
        modes_exact = modes_exact[:, order]
        modes_projected = modes_projected[:, order]
        amplitudes = amplitudes[order]
    elif sort_key in {"abs_freq", "abs_frequency", "afreq", "absolute_frequency"}:
        abs_freq = np.abs(np.angle(eigenvalues)) / (2.0 * np.pi * dt)
        order = np.argsort(abs_freq)
        eigenvalues = eigenvalues[order]
        modes_exact = modes_exact[:, order]
        modes_projected = modes_projected[:, order]
    elif sort_key != "none":
        raise ValueError("sort_modes must be 'none', 'abs_freq', or 'optimal'.")

    if return_amplitudes:
        return eigenvalues, modes_exact, modes_projected, amplitudes
    return eigenvalues, modes_exact, modes_projected
