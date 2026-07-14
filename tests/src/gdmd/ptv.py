"""Utilities for the experimental PTV rectangular-jet example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .gaps import compute_gap_snapshot_pair


def load_ptv_pair(index: int, *, field_root: str | Path, y_axis: float = 0.0, mirror_y: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load one denoised PTV particle-position pair from ``field_data_denoised``.

    The expected file is ``{index:04d}.npy`` with shape approximately
    ``(n_particles, 2, 2)``: particle index, pulse within pair, coordinate.
    """

    path = Path(field_root) / f"{index:04d}.npy"
    seeds = np.unique(np.load(path), axis=0)
    if seeds.ndim != 3 or seeds.shape[1] < 2 or seeds.shape[2] < 2:
        raise ValueError(f"Unexpected seed array shape in {path}: {seeds.shape}")
    seeds = seeds[:, :2, :2].astype(float, copy=True)
    if mirror_y:
        seeds[:, :, 1] = 2.0 * y_axis - seeds[:, :, 1]
    return np.squeeze(seeds[:, 0, :]), np.squeeze(seeds[:, 1, :])


def find_ptv_borders(*, field_root: str | Path, num_pairs: int) -> dict[str, np.ndarray]:
    """Find the spatial bounding box of the denoised PTV particle data."""

    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf
    nptmax = 0
    for i in tqdm(range(num_pairs), desc="PTV borders"):
        path = Path(field_root) / f"{i:04d}.npy"
        seeds = np.load(path)
        xmin = min(xmin, float(np.min(seeds[:, :, 0])))
        xmax = max(xmax, float(np.max(seeds[:, :, 0])))
        ymin = min(ymin, float(np.min(seeds[:, :, 1])))
        ymax = max(ymax, float(np.max(seeds[:, :, 1])))
        nptmax = max(nptmax, int(seeds.shape[0]))
    return {"x": np.asarray([xmin, xmax]), "y": np.asarray([ymin, ymax]), "npt": np.asarray(nptmax)}


def custom_y_grid_from_reference(
    y_linear: np.ndarray,
    *,
    y_axis: float,
    y_lip: float,
    ny: int,
) -> np.ndarray:
    """Construct the lip-refined y grid used by the PTV post-processing."""

    y_lip2 = 2.0 * y_axis - y_lip
    dylip = (y_axis - y_lip) / 2.0
    pchip_pts = np.asarray([y_axis, y_lip2 - dylip, y_lip2, y_lip2 + dylip, 2.0 * y_axis, y_linear.max()])
    pchip_dy = np.asarray([4.0, 4.0, 1.0, 4.0, 4.0, 6.0])
    interpolator = PchipInterpolator(pchip_pts, pchip_dy)

    def target_dy(y):
        return interpolator(y)

    def get_y(scale: float) -> np.ndarray:
        vals = [float(y_axis)]
        for _ in range(ny):
            vals.append(vals[-1] + float(scale) * float(target_dy(vals[-1])))
        return np.asarray(vals)

    def loss(scale_arr: np.ndarray) -> float:
        return abs(get_y(float(scale_arr[0]))[-1] - y_linear.max())

    result = minimize(loss, np.asarray([1.0]), method="SLSQP")
    y_half = get_y(float(result.x[0]))
    y_half_reflected = y_axis - (y_half[1:] - y_axis)[::-1]
    return np.concatenate(([0.0], y_half_reflected[y_half_reflected > 0.0], y_half))


def estimate_axis_and_lip(XX: np.ndarray, YY: np.ndarray, U: np.ndarray, *, axis_threshold: float = 2.0) -> tuple[float, float]:
    """Estimate jet centreline and lip location from the mean streamwise velocity."""

    xvec = XX[0, :]
    yvec = YY[:, 0]
    active = U > axis_threshold
    if not np.any(active):
        # Fallback for rescaled data: use a relative threshold.
        active = U > 0.1 * np.nanmax(U)
    y_axis_guess = float(np.average(YY[active]))
    lip_band = (YY < y_axis_guess) & (U < U.max() * 2.0 / 3.0) & (U > U.max() / 3.0)

    y_axis = 0.0
    y_lip = 0.0
    n_used = 0
    for ix in range(xvec.size):
        col_active = active[:, ix]
        col_lip = lip_band[:, ix]
        if np.count_nonzero(col_active) > 2:
            uint = cumulative_trapezoid(U[col_active, ix], yvec[col_active])
            yint = 0.5 * (yvec[col_active][1:] + yvec[col_active][:-1])
            if uint.size and np.isfinite(uint[-1]) and uint[-1] != 0.0:
                y_axis += float(np.interp(uint[-1] / 2.0, uint, yint))
                n_used += 1
        if np.count_nonzero(col_lip) > 2:
            try:
                y_lip += float(np.interp(U[:, ix].max() / 2.0, U[:, ix][col_lip], yvec[col_lip]))
            except Exception:
                pass
    if n_used == 0:
        return y_axis_guess, float(np.average(YY[lip_band])) if np.any(lip_band) else y_axis_guess
    return y_axis / n_used, y_lip / max(n_used, 1)


def compute_ptv_mean_velocity(
    *,
    field_root: str | Path,
    num_pairs: int,
    xvec: np.ndarray,
    yvec: np.ndarray,
    n_neighbors: int = 5,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Interpolate all PTV particle velocities onto a Cartesian grid and average."""

    XX, YY = np.meshgrid(xvec, yvec, indexing="xy")
    xy_cart = np.column_stack((XX.ravel(), YY.ravel()))
    uv_cart = np.zeros((xy_cart.shape[0], 2), dtype=float)
    knn = NearestNeighbors(n_neighbors=n_neighbors)

    for i in tqdm(range(num_pairs), desc="PTV mean velocity"):
        s1, s2 = load_ptv_pair(i, field_root=field_root)
        uv_it = (s2 - s1) / dt
        xy_it = 0.5 * (s1 + s2)
        knn.fit(xy_it)
        dist, neigh = knn.kneighbors(xy_cart)
        dist[dist < 1e-5] = 1e-5
        weights = 1.0 / dist
        uave = np.average(uv_it[neigh, 0], axis=1, weights=weights)
        vave = np.average(uv_it[neigh, 1], axis=1, weights=weights)
        uv_cart += np.column_stack((uave, vave)) / num_pairs

    U = uv_cart[:, 0].reshape(XX.shape)
    V = uv_cart[:, 1].reshape(XX.shape)
    y_axis, y_lip = estimate_axis_and_lip(XX, YY, U)
    return XX, YY, U, V, y_axis, y_lip


def compute_ptv_pair_fields(
    index: int,
    out_index: int,
    xgrid: np.ndarray,
    *,
    field_root: str | Path,
    n_neighbors_gap: int,
    n_neighbors_vel: int,
    gap_weights: str,
    dt_pair: float,
    mirror_y: bool,
    y_axis: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Compute gap fields and particle-displacement velocity for one PTV pair."""

    s1, s2 = load_ptv_pair(index, field_root=field_root, y_axis=y_axis, mirror_y=mirror_y)
    gap1, gap2 = compute_gap_snapshot_pair(
        s1,
        s2,
        xgrid,
        n_neighbors=n_neighbors_gap,
        gap_weights=gap_weights,
    )

    vp = (s2 - s1) / dt_pair
    knn_v = NearestNeighbors(n_neighbors=n_neighbors_vel)
    knn_v.fit(s1)
    _, neigh = knn_v.kneighbors(xgrid)
    vel = np.empty((xgrid.shape[0], 2), dtype=float)
    vel[:, 0] = np.mean(vp[neigh, 0], axis=1)
    vel[:, 1] = np.mean(vp[neigh, 1], axis=1)
    return out_index, gap1, gap2, vel


def compute_ptv_fields_ray(
    *,
    field_root: str | Path,
    xgrid: np.ndarray,
    num_pairs: int,
    n_neighbors_gap: int,
    n_neighbors_vel: int,
    gap_weights: str,
    dt_pair: float,
    nproc: int,
    ray_address: str | None = None,
    symmetry_augmentation: bool = False,
    y_axis: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PTV gap and velocity fields with Ray as the default backend."""

    nt = 2 * num_pairs if symmetry_augmentation else num_pairs
    density1 = np.zeros((xgrid.shape[0], 2, nt), dtype=float)
    density2 = np.zeros_like(density1)
    velocity = np.zeros_like(density1)

    tasks: list[tuple[int, int, bool]] = [(i, i, False) for i in range(num_pairs)]
    if symmetry_augmentation:
        tasks += [(i, i + num_pairs, True) for i in range(num_pairs)]

    if nproc <= 1:
        iterator = tqdm(tasks, desc="PTV fields")
        for i, out_index, mirror in iterator:
            j, g1, g2, vv = compute_ptv_pair_fields(
                i,
                out_index,
                xgrid,
                field_root=field_root,
                n_neighbors_gap=n_neighbors_gap,
                n_neighbors_vel=n_neighbors_vel,
                gap_weights=gap_weights,
                dt_pair=dt_pair,
                mirror_y=mirror,
                y_axis=y_axis,
            )
            density1[:, :, j] = g1
            density2[:, :, j] = g2
            velocity[:, :, j] = vv
        return density1, density2, velocity

    try:
        import ray
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Ray is required for parallel PTV field construction; use --nproc 1 for serial execution.") from exc

    started_ray = False
    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            num_cpus=nproc if ray_address is None else None,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        started_ray = True

    xgrid_ref = ray.put(xgrid)
    remote_worker = ray.remote(compute_ptv_pair_fields)
    futures = [
        remote_worker.remote(
            i,
            j,
            xgrid_ref,
            field_root=field_root,
            n_neighbors_gap=n_neighbors_gap,
            n_neighbors_vel=n_neighbors_vel,
            gap_weights=gap_weights,
            dt_pair=dt_pair,
            mirror_y=mirror,
            y_axis=y_axis,
        )
        for i, j, mirror in tasks
    ]

    try:
        with tqdm(total=len(futures), desc="PTV fields") as pbar:
            while futures:
                done, futures = ray.wait(futures, num_returns=1)
                j, g1, g2, vv = ray.get(done[0])
                density1[:, :, j] = g1
                density2[:, :, j] = g2
                velocity[:, :, j] = vv
                pbar.update(1)
    finally:
        if started_ray:
            ray.shutdown()

    return density1, density2, velocity
