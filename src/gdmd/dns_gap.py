"""Ray-backed gap-field construction for DNS-generated particle pairs."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .gaps import compute_gap_snapshot_pair


def _dns_gap_worker(
    i: int,
    seeds1: np.ndarray,
    seeds2: np.ndarray,
    xgrid: np.ndarray,
    n_neighbors: int,
    gap_weights: str,
) -> tuple[int, np.ndarray, np.ndarray]:
    gap1, gap2 = compute_gap_snapshot_pair(
        seeds1[:, :, i],
        seeds2[:, :, i],
        xgrid,
        n_neighbors=n_neighbors,
        gap_weights=gap_weights,
    )
    return i, gap1, gap2


def compute_dns_gap_fields(
    seeds1: np.ndarray,
    seeds2: np.ndarray,
    xgrid: np.ndarray,
    *,
    n_neighbors: int,
    gap_weights: str,
    nproc: int,
    ray_address: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Eulerian gap fields for DNS-generated particle-pair arrays."""

    seeds1 = np.asarray(seeds1)
    seeds2 = np.asarray(seeds2)
    xgrid = np.asarray(xgrid)
    if seeds1.shape != seeds2.shape:
        raise ValueError("seeds1 and seeds2 must have the same shape.")
    if seeds1.ndim != 3 or seeds1.shape[1] != 2:
        raise ValueError("seed arrays must have shape (n_seeds, 2, n_snapshots).")

    n_snapshots = seeds1.shape[2]
    density1 = np.zeros((xgrid.shape[0], 2, n_snapshots), dtype=float)
    density2 = np.zeros_like(density1)

    if nproc <= 1:
        for i in tqdm(range(n_snapshots), desc="Gap fields"):
            density1[:, :, i], density2[:, :, i] = compute_gap_snapshot_pair(
                seeds1[:, :, i],
                seeds2[:, :, i],
                xgrid,
                n_neighbors=n_neighbors,
                gap_weights=gap_weights,
            )
        return density1, density2

    try:
        import ray
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Ray is required for parallel gap-field construction; use --nproc 1 for serial execution.") from exc

    started_ray = False
    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            num_cpus=nproc if ray_address is None else None,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        started_ray = True

    seeds1_ref = ray.put(seeds1)
    seeds2_ref = ray.put(seeds2)
    xgrid_ref = ray.put(xgrid)
    worker = ray.remote(_dns_gap_worker)
    futures = [
        worker.remote(i, seeds1_ref, seeds2_ref, xgrid_ref, n_neighbors, gap_weights)
        for i in range(n_snapshots)
    ]

    try:
        with tqdm(total=n_snapshots, desc="Gap fields") as pbar:
            while futures:
                done, futures = ray.wait(futures, num_returns=1)
                i, gap1, gap2 = ray.get(done[0])
                density1[:, :, i] = gap1
                density2[:, :, i] = gap2
                pbar.update(1)
    finally:
        if started_ray:
            ray.shutdown()

    return density1, density2
