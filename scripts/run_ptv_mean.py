#!/usr/bin/env python3
"""Prepare the mean velocity grid for the experimental PTV rectangular-jet case.

The recommended workflow is two-pass:

1. Build a linear grid and estimate the jet centreline/lip:
   ``python scripts/run_ptv_mean.py --grid linear --output data/meanvel_linspace.npz``
2. Build the lip-refined grid using the first-pass result:
   ``python scripts/run_ptv_mean.py --grid custom --reference-mean data/meanvel_linspace.npz --output data/meanvel_customspace.npz``
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gdmd import compute_ptv_mean_velocity, custom_y_grid_from_reference, find_ptv_borders  # noqa: E402


@dataclass
class Config:
    field_root: Path = Path("field_data_denoised")
    num_pairs: int = 1198
    output: Path = Path("data/meanvel_customspace.npz")
    borders: Path = Path("data/borders.npz")
    recompute_borders: bool = False
    grid: str = "custom"
    reference_mean: Path = Path("data/meanvel_linspace.npz")
    nx: int = 180
    ny: int = 60
    n_neighbors: int = 5
    dt: float = 1.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-root", type=Path, default=Config.field_root, help="Directory containing {i:04d}.npy denoised PTV pairs.")
    parser.add_argument("--num-pairs", type=int, default=Config.num_pairs)
    parser.add_argument("--output", type=Path, default=Config.output)
    parser.add_argument("--borders", type=Path, default=Config.borders)
    parser.add_argument("--recompute-borders", action=argparse.BooleanOptionalAction, default=Config.recompute_borders)
    parser.add_argument("--grid", choices=("linear", "custom"), default=Config.grid)
    parser.add_argument("--reference-mean", type=Path, default=Config.reference_mean, help="Reference mean file used to build the custom y grid.")
    parser.add_argument("--nx", type=int, default=Config.nx)
    parser.add_argument("--ny", type=int, default=Config.ny)
    parser.add_argument("--n-neighbors", type=int, default=Config.n_neighbors)
    parser.add_argument("--dt", type=float, default=Config.dt, help="Pulse-pair time delay used to compute particle displacement velocity.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.borders.parent.mkdir(parents=True, exist_ok=True)

    if args.recompute_borders or not args.borders.exists():
        borders = find_ptv_borders(field_root=args.field_root, num_pairs=args.num_pairs)
        np.savez(args.borders, **borders)
        print(f"Saved {args.borders}")
    else:
        borders_npz = np.load(args.borders)
        borders = {"x": borders_npz["x"], "y": borders_npz["y"], "npt": borders_npz["npt"]}

    xvec = np.linspace(float(np.min(borders["x"])), float(np.max(borders["x"])), args.nx)
    y_linear = np.linspace(float(np.min(borders["y"])), float(np.max(borders["y"])), args.ny)

    if args.grid == "linear":
        yvec = y_linear
    else:
        if not args.reference_mean.exists():
            raise FileNotFoundError(
                f"{args.reference_mean} does not exist. Run once with --grid linear first, "
                "or provide --reference-mean."
            )
        ref = np.load(args.reference_mean)
        yvec = custom_y_grid_from_reference(
            y_linear,
            y_axis=float(ref["yaxis"]),
            y_lip=float(ref["ylip"]),
            ny=args.ny,
        )

    XX, YY, U, V, y_axis, y_lip = compute_ptv_mean_velocity(
        field_root=args.field_root,
        num_pairs=args.num_pairs,
        xvec=xvec,
        yvec=yvec,
        n_neighbors=args.n_neighbors,
        dt=args.dt,
    )

    np.savez(args.output, X=XX, Y=YY, U=U, V=V, yaxis=y_axis, ylip=y_lip)
    print(f"Saved {args.output}")
    print(f"Estimated y_axis = {y_axis:.6g}, y_lip = {y_lip:.6g}")


if __name__ == "__main__":
    main()
