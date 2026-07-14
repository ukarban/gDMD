"""Shared exact-DMD/gDMD analysis routine for all examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dmd import dmd_pair
from .mode_matching import match_dmd_modes
from .weighting import apply_spatial_weights, tke_weights_from_velocity, undo_spatial_weights


@dataclass(frozen=True)
class GDMDAnalysisOptions:
    """Options for the shared exact-DMD/gDMD comparison."""

    n_modes: int
    dt_exact: float
    dt_gap: float
    use_tke_weighting: bool = False
    tke_weight_alpha: float = 0.5
    tke_weight_floor: float = 0.05
    match_w_eig: float = 0.4
    match_w_mode: float = 0.6
    weighted_match_w_eig: float | None = None
    weighted_match_w_mode: float | None = None
    exact_filter_min_real: float | None = None
    continuous_mode_matching: bool = False
    positive_frequency_matching: bool = False
    include_conjugate_partners: bool = False
    center_exact: bool = True
    center_gap: bool = True


def _slice_columns(X: np.ndarray, indices: np.ndarray | slice | None) -> np.ndarray:
    if indices is None:
        return X
    return X[:, indices]


def run_gdmd_analysis(
    *,
    exact1: np.ndarray,
    exact2: np.ndarray,
    gap1: np.ndarray,
    gap2: np.ndarray,
    n_points: int,
    options: GDMDAnalysisOptions,
    exact_indices: np.ndarray | slice | None = None,
    gap_indices: np.ndarray | slice | None = None,
) -> dict[str, np.ndarray]:
    """Run exact DMD, baseline gDMD, and optional TKE-weighted gDMD.

    All three public drivers call this function, so the DMD/gDMD decomposition
    follows the same ordering, weighting, matching, and output-key conventions in
    the cylinder, pinball, and experimental PTV cases.
    """

    X1 = _slice_columns(np.asarray(exact1), exact_indices)
    X2 = _slice_columns(np.asarray(exact2), exact_indices)
    G1 = _slice_columns(np.asarray(gap1), gap_indices)
    G2 = _slice_columns(np.asarray(gap2), gap_indices)

    if X1.shape != X2.shape:
        raise ValueError("exact1 and exact2 must have the same selected shape.")
    if G1.shape != G2.shape:
        raise ValueError("gap1 and gap2 must have the same selected shape.")
    if X1.shape[0] != G1.shape[0]:
        raise ValueError("exact and gap state dimensions must match.")
    if X1.shape[0] != 2 * n_points:
        raise ValueError("n_points is inconsistent with the state dimension.")

    E_exact, Phi_exact, Phi_exact_proj = dmd_pair(
        X1,
        X2,
        options.n_modes,
        options.dt_exact,
        center=options.center_exact,
        sort_modes="abs_freq",
    )

    if options.exact_filter_min_real is not None:
        keep = np.real(E_exact) > options.exact_filter_min_real
        E_exact = E_exact[keep]
        Phi_exact = Phi_exact[:, keep]
        Phi_exact_proj = Phi_exact_proj[:, keep]

    E_gap, Phi_gap, Phi_gap_proj = dmd_pair(
        G1,
        G2,
        options.n_modes,
        options.dt_gap,
        center=options.center_gap,
        sort_modes="abs_freq",
    )
    Phi_gdmd = Phi_gap - Phi_gap_proj

    order, Phi_gdmd_aligned, iref, S, S_eig, S_mode, phases = match_dmd_modes(
        E_exact,
        Phi_exact,
        E_gap,
        Phi_gdmd,
        w_eig=options.match_w_eig,
        w_mode=options.match_w_mode,
        dt_ref=options.dt_exact,
        dt_cmp=options.dt_gap,
        continuous=options.continuous_mode_matching,
        positive_frequency_only=options.positive_frequency_matching,
        include_conjugate_partners=options.include_conjugate_partners,
    )

    results: dict[str, np.ndarray] = {
        "E_exact": E_exact,
        "Phi_exact": Phi_exact,
        "Phi_exact_projected": Phi_exact_proj,
        "E_exact_matched": E_exact[iref],
        "Phi_exact_matched": Phi_exact[:, iref],
        "E_gdmd": E_gap[order],
        "Phi_gdmd": Phi_gdmd_aligned[:, order],
        "Phi_gap_exact": Phi_gap[:, order],
        "Phi_gap_projected": Phi_gap_proj[:, order],
        "exact_match_order": iref,
        "gdmd_match_order": order,
        "match_score": S,
        "match_score_eigenvalue": S_eig,
        "match_score_mode": S_mode,
        "match_phases": phases,
        "dmd_dt_exact": np.asarray(options.dt_exact),
        "dmd_dt_gap": np.asarray(options.dt_gap),
    }

    if options.use_tke_weighting:
        w_tke = tke_weights_from_velocity(
            X1,
            X2,
            n_points,
            alpha=options.tke_weight_alpha,
            floor=options.tke_weight_floor,
        )
        G1w, sqrtw = apply_spatial_weights(G1, w_tke)
        G2w, _ = apply_spatial_weights(G2, w_tke)
        E_gap_w, Phi_gap_w_s, Phi_gap_proj_w_s = dmd_pair(
            G1w,
            G2w,
            options.n_modes,
            options.dt_gap,
            center=options.center_gap,
            sort_modes="abs_freq",
        )
        Phi_gap_w = undo_spatial_weights(Phi_gap_w_s, sqrtw)
        Phi_gap_proj_w = undo_spatial_weights(Phi_gap_proj_w_s, sqrtw)
    else:
        w_tke = np.ones(n_points)
        E_gap_w = E_gap
        Phi_gap_w = Phi_gap
        Phi_gap_proj_w = Phi_gap_proj

    Phi_gdmd_w = Phi_gap_w - Phi_gap_proj_w
    order_w, Phi_gdmd_w_aligned, iref_w, S_w, S_eig_w, S_mode_w, phases_w = match_dmd_modes(
        E_exact,
        Phi_exact,
        E_gap_w,
        Phi_gdmd_w,
        w_eig=options.weighted_match_w_eig if options.weighted_match_w_eig is not None else options.match_w_eig,
        w_mode=options.weighted_match_w_mode if options.weighted_match_w_mode is not None else options.match_w_mode,
        dt_ref=options.dt_exact,
        dt_cmp=options.dt_gap,
        continuous=options.continuous_mode_matching,
        positive_frequency_only=options.positive_frequency_matching,
        include_conjugate_partners=options.include_conjugate_partners,
    )

    results.update(
        {
            "E_exact_weighted_matched": E_exact[iref_w],
            "Phi_exact_weighted_matched": Phi_exact[:, iref_w],
            "E_gdmd_weighted": E_gap_w[order_w],
            "Phi_gdmd_weighted": Phi_gdmd_w_aligned[:, order_w],
            "Phi_gap_exact_weighted": Phi_gap_w[:, order_w],
            "Phi_gap_projected_weighted": Phi_gap_proj_w[:, order_w],
            "exact_weighted_match_order": iref_w,
            "gdmd_weighted_match_order": order_w,
            "weighted_match_score": S_w,
            "weighted_match_score_eigenvalue": S_eig_w,
            "weighted_match_score_mode": S_mode_w,
            "weighted_match_phases": phases_w,
            "tke_weight": w_tke,
            "tke_weight_alpha": np.asarray(options.tke_weight_alpha),
            "tke_weight_floor": np.asarray(options.tke_weight_floor),
        }
    )
    return results
