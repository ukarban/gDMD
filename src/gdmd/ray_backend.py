"""Ray helpers used by the gDMD drivers.

Ray is intentionally the default backend for the public scripts because the
snapshot-level gap construction is expensive and the large arrays can be shared
through Ray's object store.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from tqdm import tqdm


def run_ray_tasks(
    tasks: Iterable[Any],
    worker_fn: Callable[..., Any],
    *,
    nproc: int,
    ray_address: str | None = None,
    desc: str = "Ray tasks",
) -> list[Any]:
    """Run independent tasks with Ray and return results in completion order.

    The function deliberately does not implement caching. It is meant for clean,
    one-pass reproduction runs.
    """

    if nproc <= 1:
        return [worker_fn(*task) if isinstance(task, tuple) else worker_fn(task) for task in tqdm(list(tasks), desc=desc)]

    try:
        import ray
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Ray is required for parallel execution. Install the package "
            "environment or rerun the script with --nproc 1."
        ) from exc

    started_ray = False
    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            num_cpus=nproc if ray_address is None else None,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        started_ray = True

    remote_worker = ray.remote(worker_fn)
    futures = [remote_worker.remote(*task) if isinstance(task, tuple) else remote_worker.remote(task) for task in tasks]
    out: list[Any] = []

    try:
        with tqdm(total=len(futures), desc=desc) as pbar:
            while futures:
                done, futures = ray.wait(futures, num_returns=1)
                out.append(ray.get(done[0]))
                pbar.update(1)
    finally:
        if started_ray:
            ray.shutdown()

    return out
