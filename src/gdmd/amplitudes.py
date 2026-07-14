"""Optional DMD amplitude analysis utilities."""

from __future__ import annotations

import numpy as np


def field_scale_l2_rms(X: np.ndarray, *, eps: float = 1e-14) -> float:
    """Return the RMS L2 norm of the snapshot columns."""

    return float(np.sqrt(np.mean(np.linalg.norm(X, axis=0) ** 2)) + eps)


def select_uv_component(X: np.ndarray, Phi: np.ndarray, component: str = "both"):
    """Select the streamwise component, transverse component, or both."""

    component = component.lower()
    if component == "both":
        idx = np.arange(Phi.shape[0])
    elif component == "u":
        idx = np.arange(0, Phi.shape[0], 2)
    elif component == "v":
        idx = np.arange(1, Phi.shape[0], 2)
    else:
        raise ValueError("component must be 'both', 'u', or 'v'.")
    return X[idx, :], Phi[idx, :], idx


def dmd_amp_analysis(
    X: np.ndarray,
    Phi: np.ndarray,
    eigs: np.ndarray | None = None,
    *,
    component: str = "both",
    dt: float = 1.0,
    fmin_amp: float = 0.0,
    rcond: float = 1e-12,
    eps: float = 1e-14,
) -> dict:
    """Compute RMS, classical, and optimal DMD amplitudes."""

    Xc, Phic, row_idx = select_uv_component(X, Phi, component=component)
    n_snap = Xc.shape[1]

    if eigs is None:
        active = np.ones(Phic.shape[1], dtype=bool)
        eigs_a = None
    else:
        freq = np.angle(eigs) / (2.0 * np.pi * dt)
        active = np.abs(freq) >= fmin_amp
        eigs_a = eigs[active]

    Phi_a = Phic[:, active]
    X_scale = field_scale_l2_rms(Xc, eps=eps)

    mode_norms = np.linalg.norm(Phi_a, axis=0)
    mode_norms = np.where(mode_norms > eps, mode_norms, 1.0)
    Phi_n = Phi_a / mode_norms[None, :]

    out = {
        "component": component,
        "row_idx": row_idx,
        "active": active,
        "Phi_n": Phi_n,
        "mode_norms": mode_norms,
        "field_scale": X_scale,
        "amp": {},
        "amp_rel": {},
        "err": {},
    }

    A, *_ = np.linalg.lstsq(Phi_n, Xc, rcond=rcond)
    Xr = Phi_n @ A
    a_rms = np.sqrt(np.mean(np.abs(A) ** 2, axis=1))
    out["amp"]["rms"] = a_rms
    out["amp_rel"]["rms"] = a_rms / X_scale
    out["err"]["rms"] = np.linalg.norm(Xc - Xr, axis=0) / (np.linalg.norm(Xc, axis=0) + eps)

    if eigs_a is not None:
        k = np.arange(n_snap)
        V = eigs_a[:, None] ** k[None, :]
        out["V"] = V

        b_classical, *_ = np.linalg.lstsq(Phi_n, Xc[:, 0], rcond=rcond)
        Xr = Phi_n @ (b_classical[:, None] * V)
        out["amp"]["classical"] = b_classical
        out["amp_rel"]["classical"] = b_classical / X_scale
        out["err"]["classical"] = np.linalg.norm(Xc - Xr, axis=0) / (np.linalg.norm(Xc, axis=0) + eps)

        G_phi = Phi_n.conj().T @ Phi_n
        G_vand = V.conj() @ V.T
        H = G_phi * G_vand
        rhs = np.sum((Phi_n.conj().T @ Xc) * V.conj(), axis=1)
        b_opt, *_ = np.linalg.lstsq(H, rhs, rcond=rcond)
        Xr = Phi_n @ (b_opt[:, None] * V)
        out["amp"]["optimal"] = b_opt
        out["amp_rel"]["optimal"] = b_opt / X_scale
        out["err"]["optimal"] = np.linalg.norm(Xc - Xr, axis=0) / (np.linalg.norm(Xc, axis=0) + eps)

    return out
