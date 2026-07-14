"""Mode matching utilities shared by all gDMD examples."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _finite_scale(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 1.0
    scale = float(np.median(vals))
    return max(scale, 1.0e-6)


def _conjugate_partner(eigs: np.ndarray, index: int) -> int:
    return int(np.argmin(np.abs(eigs - np.conj(eigs[index]))))


def _unique_pairs(ref_indices: np.ndarray, cmp_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep: list[tuple[int, int]] = []
    seen_ref: set[int] = set()
    seen_cmp: set[int] = set()
    for ir, ic in zip(ref_indices, cmp_indices, strict=False):
        ir_i = int(ir)
        ic_i = int(ic)
        if ir_i not in seen_ref and ic_i not in seen_cmp:
            keep.append((ir_i, ic_i))
            seen_ref.add(ir_i)
            seen_cmp.add(ic_i)
    if not keep:
        return np.array([], dtype=int), np.array([], dtype=int)
    out_ref = np.array([p[0] for p in keep], dtype=int)
    out_cmp = np.array([p[1] for p in keep], dtype=int)
    order = np.argsort(out_ref)
    return out_ref[order], out_cmp[order]


def match_dmd_modes(
    Eref: np.ndarray,
    Phiref: np.ndarray,
    Ecmp: np.ndarray,
    Phicmp: np.ndarray,
    *,
    w_eig: float = 0.4,
    w_mode: float = 0.6,
    dt_ref: float = 1.0,
    dt_cmp: float = 1.0,
    continuous: bool = False,
    positive_frequency_only: bool = False,
    include_conjugate_partners: bool = False,
    eps: float = 1e-14,
):
    """Match comparison DMD modes to reference modes and phase-align them.

    Parameters
    ----------
    Eref, Phiref:
        Reference eigenvalues and modes, usually from exact DMD.
    Ecmp, Phicmp:
        Comparison eigenvalues and modes, usually from gDMD.
    continuous:
        If ``False``, eigenvalue matching is based on discrete eigenvalue radius
        and angle. If ``True``, matching is based on the continuous-time growth
        rates and angular frequencies, ``log(lambda)/dt``. The latter is useful
        when exact DMD and gDMD use different time intervals.
    positive_frequency_only:
        If ``True``, solve the assignment problem only over positive-frequency
        modes. This is primarily used for the experimental PTV case.
    include_conjugate_partners:
        If ``True``, add the conjugate partner of every positive-frequency match.

    Returns
    -------
    cmp_order, Phicmp_aligned, ref_order, S, S_eig, S_mode, phases
        ``Phicmp_aligned`` has the same shape as ``Phicmp``. The selected matched
        comparison modes are obtained as ``Phicmp_aligned[:, cmp_order]`` and
        correspond to ``Phiref[:, ref_order]``.
    """

    Eref = np.asarray(Eref)
    Ecmp = np.asarray(Ecmp)
    Phiref = np.asarray(Phiref)
    Phicmp = np.asarray(Phicmp)

    if Phiref.shape[0] != Phicmp.shape[0]:
        raise ValueError("Reference and comparison modes must have the same state dimension.")
    if Eref.size != Phiref.shape[1] or Ecmp.size != Phicmp.shape[1]:
        raise ValueError("Eigenvalue counts must match the number of mode columns.")

    if continuous:
        wref = np.log(Eref) / dt_ref
        wcmp = np.log(Ecmp) / dt_cmp
        candidate_ref = np.arange(Eref.size)
        candidate_cmp = np.arange(Ecmp.size)
        if positive_frequency_only:
            candidate_ref = np.where(np.imag(wref) > eps)[0]
            candidate_cmp = np.where(np.imag(wcmp) > eps)[0]
        wr = wref[candidate_ref]
        wc = wcmp[candidate_cmp]
        dg = np.abs(np.real(wr[:, None]) - np.real(wc[None, :]))
        df = np.abs(np.imag(wr[:, None]) - np.imag(wc[None, :]))
        sg = _finite_scale(dg)
        sf = _finite_scale(df)
        S_eig = np.exp(-(dg / sg) ** 2) * np.exp(-(df / sf) ** 2)
    else:
        candidate_ref = np.arange(Eref.size)
        candidate_cmp = np.arange(Ecmp.size)
        if positive_frequency_only:
            candidate_ref = np.where(np.angle(Eref) > eps)[0]
            candidate_cmp = np.where(np.angle(Ecmp) > eps)[0]
        Er = Eref[candidate_ref]
        Ec = Ecmp[candidate_cmp]
        dr = np.abs(
            np.log(np.maximum(np.abs(Er[:, None]), eps))
            - np.log(np.maximum(np.abs(Ec[None, :]), eps))
        )
        da = np.abs(np.angle(Er[:, None]) - np.angle(Ec[None, :]))
        da = np.minimum(da, 2.0 * np.pi - da)
        sr = _finite_scale(dr)
        sa = _finite_scale(da)
        # Keep the radius term weak by default, as in the original DNS scripts.
        S_eig = np.exp(-(dr / sr / 100.0) ** 2) * np.exp(-(da / sa) ** 2)

    if candidate_ref.size == 0 or candidate_cmp.size == 0:
        raise RuntimeError("No candidate modes are available for matching.")

    Pr = Phiref[:, candidate_ref]
    Pc = Phicmp[:, candidate_cmp]
    Prn = Pr / np.maximum(np.linalg.norm(Pr, axis=0, keepdims=True), eps)
    Pcn = Pc / np.maximum(np.linalg.norm(Pc, axis=0, keepdims=True), eps)
    S_mode = np.abs(Prn.conj().T @ Pcn)
    S = w_eig * S_eig + w_mode * S_mode

    local_ref, local_cmp = linear_sum_assignment(-S)
    sort_order = np.argsort(local_ref)
    local_ref = local_ref[sort_order]
    local_cmp = local_cmp[sort_order]
    ref_order = candidate_ref[local_ref]
    cmp_order = candidate_cmp[local_cmp]

    if include_conjugate_partners:
        ref_ext: list[int] = []
        cmp_ext: list[int] = []
        for ir, ic in zip(ref_order, cmp_order, strict=False):
            ref_ext.extend([int(ir), _conjugate_partner(Eref, int(ir))])
            cmp_ext.extend([int(ic), _conjugate_partner(Ecmp, int(ic))])
        ref_order, cmp_order = _unique_pairs(np.array(ref_ext, dtype=int), np.array(cmp_ext, dtype=int))
    else:
        ref_order, cmp_order = _unique_pairs(ref_order, cmp_order)

    Phicmp_aligned = Phicmp.copy()
    phases = np.ones(cmp_order.size, dtype=complex)
    for k, (ir, ic) in enumerate(zip(ref_order, cmp_order, strict=False)):
        alpha = np.vdot(Phiref[:, ir], Phicmp[:, ic])
        phase = 1.0 if np.abs(alpha) < eps else np.exp(-1j * np.angle(alpha))
        Phicmp_aligned[:, ic] *= phase
        phases[k] = phase

    return cmp_order, Phicmp_aligned, ref_order, S, S_eig, S_mode, phases
