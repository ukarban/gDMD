#!/usr/bin/env python3
"""Run exact DMD and gDMD for the experimental PTV rectangular-jet dataset."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gdmd import (  # noqa: E402
    GDMDAnalysisOptions,
    compute_ptv_fields_ray,
    rectangular_window_mask,
    restrict_state_matrix,
    run_gdmd_analysis,
)


@dataclass
class Config:
    field_root: Path = Path("field_data_denoised")
    mean_file: Path = Path("data/meanvel_customspace.npz")
    output_file: Path = Path("results/ptv_exact_vs_gdmd_results.npz")
    num_pairs: int = 1198
    fs: float = 10_000.0
    normalized_time: bool = False
    n_modes: int = 8
    n_neighbors_gap: int = 30
    n_neighbors_vel: int = 30
    gap_weights: str = "inv_r"
    xlim: tuple[float, float] = (100.0, 450.0)
    y_start: float = 50.0
    start_fraction: float = 0.0
    nproc: int = max(1, (os.cpu_count() or 2) - 1)
    ray_address: str | None = None
    symmetry_augmentation: bool = False
    use_tke_weighting: bool = False
    tke_weight_alpha: float = 0.5
    tke_weight_floor: float = 0.05
    exact_filter_min_real: float = 0.95
    save_figures: Path | None = None
    show: bool = False
    mode_to_plot: int = 1
    component_to_plot: int = 0

    @property
    def dt_pair(self) -> float:
        return 1.0 if self.normalized_time else 1.0 / self.fs

    @property
    def dt_exact(self) -> float:
        return self.dt_pair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-root", type=Path, default=Config.field_root)
    parser.add_argument("--mean-file", type=Path, default=Config.mean_file)
    parser.add_argument("--output", dest="output_file", type=Path, default=Config.output_file)
    parser.add_argument("--num-pairs", type=int, default=Config.num_pairs)
    parser.add_argument("--fs", type=float, default=Config.fs, help="Acquisition frequency in Hz. Ignored when --normalized-time is used.")
    parser.add_argument("--normalized-time", action=argparse.BooleanOptionalAction, default=Config.normalized_time, help="Use dt=1 rather than the physical acquisition time.")
    parser.add_argument("--n-modes", type=int, default=Config.n_modes)
    parser.add_argument("--n-neighbors-gap", type=int, default=Config.n_neighbors_gap)
    parser.add_argument("--n-neighbors-vel", type=int, default=Config.n_neighbors_vel)
    parser.add_argument("--gap-weights", choices=("uniform", "inv_r", "gaussian"), default=Config.gap_weights)
    parser.add_argument("--xlim", type=float, nargs=2, default=Config.xlim)
    parser.add_argument("--y-start", type=float, default=Config.y_start, help="Lower y limit. Upper limit is 2*y_axis - y_start.")
    parser.add_argument("--start-fraction", type=float, default=Config.start_fraction)
    parser.add_argument("--nproc", type=int, default=Config.nproc, help="Ray workers; use 1 for serial execution.")
    parser.add_argument("--ray-address", type=str, default=Config.ray_address)
    parser.add_argument("--symmetry-augmentation", action=argparse.BooleanOptionalAction, default=Config.symmetry_augmentation, help="Append a mirrored copy about the estimated jet centreline.")
    parser.add_argument("--use-tke-weighting", action=argparse.BooleanOptionalAction, default=Config.use_tke_weighting)
    parser.add_argument("--tke-weight-alpha", type=float, default=Config.tke_weight_alpha)
    parser.add_argument("--tke-weight-floor", type=float, default=Config.tke_weight_floor)
    parser.add_argument("--exact-filter-min-real", type=float, default=Config.exact_filter_min_real)
    parser.add_argument("--save-figures", type=Path, default=Config.save_figures)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=Config.show)
    parser.add_argument("--mode-to-plot", type=int, default=Config.mode_to_plot, help="One-based matched-mode index for the optional mode figure.")
    parser.add_argument("--component-to-plot", type=int, choices=(0, 1), default=Config.component_to_plot)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    values = vars(args).copy()
    values["xlim"] = tuple(values["xlim"])
    return Config(**values)


def load_mean_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    mean = np.load(path)
    XX = np.asarray(mean["X"])
    YY = np.asarray(mean["Y"])
    y_axis = float(mean["yaxis"])
    y_lip = float(mean["ylip"])
    keep_y = YY[:, 0] <= 2.0 * y_axis
    XX = XX[keep_y, :]
    YY = YY[keep_y, :]
    xgrid = np.column_stack((XX.ravel(), YY.ravel()))
    return XX, YY, xgrid, y_axis, y_lip


def ptv_exact_and_gap_indices(num_pairs: int, nt_total: int, start_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n0 = int(start_fraction * (num_pairs - 1))
    if nt_total == num_pairs:
        exact = np.arange(n0, num_pairs - 1)
        gap = np.arange(n0, num_pairs)
    else:
        exact = np.concatenate((np.arange(n0, num_pairs - 1), np.arange(n0 + num_pairs, nt_total - 1)))
        gap = np.concatenate((np.arange(n0, num_pairs), np.arange(n0 + num_pairs, nt_total)))
    return exact, gap


def run_case(cfg: Config) -> dict[str, np.ndarray]:
    XX, YY, xgrid, y_axis, y_lip = load_mean_grid(cfg.mean_file)
    ylim = (cfg.y_start, 2.0 * y_axis - cfg.y_start)
    print(f"Loaded {cfg.mean_file}")
    print(f"DMD window: x={cfg.xlim}, y={ylim}")

    density1, density2, velocity = compute_ptv_fields_ray(
        field_root=cfg.field_root,
        xgrid=xgrid,
        num_pairs=cfg.num_pairs,
        n_neighbors_gap=cfg.n_neighbors_gap,
        n_neighbors_vel=cfg.n_neighbors_vel,
        gap_weights=cfg.gap_weights,
        dt_pair=cfg.dt_pair,
        nproc=cfg.nproc,
        ray_address=cfg.ray_address,
        symmetry_augmentation=cfg.symmetry_augmentation,
        y_axis=y_axis,
    )

    point_mask = rectangular_window_mask(xgrid, cfg.xlim, ylim)
    xdmd = xgrid[point_mask]
    n_points = xdmd.shape[0]
    if n_points == 0:
        raise RuntimeError("The DMD window contains no grid points.")

    gap1_all = restrict_state_matrix(density1, point_mask)
    gap2_all = restrict_state_matrix(density2, point_mask)
    vel_all = restrict_state_matrix(velocity, point_mask)
    vel_all = vel_all - vel_all.mean(axis=1, keepdims=True)

    exact_indices, gap_indices = ptv_exact_and_gap_indices(
        cfg.num_pairs,
        velocity.shape[2],
        cfg.start_fraction,
    )
    exact1 = vel_all[:, exact_indices]
    exact2 = vel_all[:, exact_indices + 1]
    gap1 = gap1_all[:, gap_indices]
    gap2 = gap2_all[:, gap_indices]

    if exact1.shape[1] < cfg.n_modes + 1 or gap1.shape[1] < cfg.n_modes + 1:
        raise RuntimeError("The selected PTV snapshot range is too short for the requested number of modes.")

    options = GDMDAnalysisOptions(
        n_modes=cfg.n_modes,
        dt_exact=cfg.dt_exact,
        dt_gap=cfg.dt_pair,
        use_tke_weighting=cfg.use_tke_weighting,
        tke_weight_alpha=cfg.tke_weight_alpha,
        tke_weight_floor=cfg.tke_weight_floor,
        match_w_eig=0.6,
        match_w_mode=0.4,
        weighted_match_w_eig=0.6,
        weighted_match_w_mode=0.4,
        exact_filter_min_real=cfg.exact_filter_min_real,
        continuous_mode_matching=True,
        positive_frequency_matching=True,
        include_conjugate_partners=True,
    )
    results = run_gdmd_analysis(
        exact1=exact1,
        exact2=exact2,
        gap1=gap1,
        gap2=gap2,
        n_points=n_points,
        options=options,
    )
    results.update(
        {
            "xdmd": xdmd,
            "point_mask": point_mask,
            "xgrid": xgrid,
            "X": XX,
            "Y": YY,
            "xlim": np.asarray(cfg.xlim),
            "ylim": np.asarray(ylim),
            "exact_indices": exact_indices,
            "gap_indices": gap_indices,
            "fs": np.asarray(cfg.fs),
            "dt_pair": np.asarray(cfg.dt_pair),
            "dt_exact": np.asarray(cfg.dt_exact),
            "y_axis": np.asarray(y_axis),
            "y_lip": np.asarray(y_lip),
            "case": np.asarray("ptv_rectangular_jet"),
        }
    )
    return results


def save_results(results: dict[str, np.ndarray], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **results)
    print(f"Saved {output_file}")


def save_or_show_figures(results: dict[str, np.ndarray], cfg: Config) -> None:
    if cfg.save_figures is None and not cfg.show:
        return
    import matplotlib.pyplot as plt

    if cfg.save_figures is not None:
        cfg.save_figures.mkdir(parents=True, exist_ok=True)

    eig_key = "E_gdmd_weighted" if cfg.use_tke_weighting else "E_gdmd"
    mode_key = "Phi_gdmd_weighted" if cfg.use_tke_weighting else "Phi_gdmd"
    exact_key = "Phi_exact_weighted_matched" if cfg.use_tke_weighting else "Phi_exact_matched"
    label = "TKE-weighted gDMD" if cfg.use_tke_weighting else "gDMD"

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(np.real(results["E_exact"]), np.imag(results["E_exact"]), "s", mfc="none", label="Exact DMD")
    ax.plot(np.real(results[eig_key]), np.imag(results[eig_key]), "^", mfc="none", label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    if cfg.save_figures is not None:
        fig.savefig(cfg.save_figures / "ptv_eigenvalues.png", dpi=300)

    k = cfg.mode_to_plot - 1
    if k < 0 or k >= results[mode_key].shape[1]:
        raise ValueError("--mode-to-plot is out of range for the matched modes.")
    component = cfg.component_to_plot
    n_points = results["xdmd"].shape[0]
    phi_exact = np.real(results[exact_key][:, k].reshape(n_points, 2)[:, component])
    phi_gdmd = np.real(results[mode_key][:, k].reshape(n_points, 2)[:, component])

    fig2, axes = plt.subplots(2, 1, figsize=(6.0, 4.5), sharex=True, sharey=True)
    for ax_i, field, title in zip(axes, (phi_exact, phi_gdmd), ("Exact DMD", label), strict=False):
        im = ax_i.tripcolor(results["xdmd"][:, 0], results["xdmd"][:, 1], field, shading="gouraud")
        ax_i.set_xlim(results["xlim"])
        ax_i.set_ylim(results["ylim"])
        ax_i.set_aspect("equal", adjustable="box")
        ax_i.set_title(title)
        fig2.colorbar(im, ax=ax_i)
    fig2.tight_layout()
    if cfg.save_figures is not None:
        fig2.savefig(cfg.save_figures / "ptv_mode_comparison.png", dpi=300)

    if cfg.use_tke_weighting:
        fig3, ax3 = plt.subplots(figsize=(5.0, 2.5))
        im3 = ax3.tripcolor(results["xdmd"][:, 0], results["xdmd"][:, 1], results["tke_weight"], shading="gouraud")
        ax3.set_xlim(results["xlim"])
        ax3.set_ylim(results["ylim"])
        ax3.set_aspect("equal", adjustable="box")
        ax3.set_title("TKE-based gDMD weight")
        fig3.colorbar(im3, ax=ax3)
        fig3.tight_layout()
        if cfg.save_figures is not None:
            fig3.savefig(cfg.save_figures / "ptv_tke_weight.png", dpi=300)

    if cfg.show:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    cfg = config_from_args(build_arg_parser().parse_args())
    results = run_case(cfg)
    save_results(results, cfg.output_file)
    save_or_show_figures(results, cfg)


if __name__ == "__main__":
    main()
