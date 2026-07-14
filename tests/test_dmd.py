import numpy as np

from gdmd.dmd import dmd_pair


def test_dmd_pair_recovers_linear_map_eigenvalues():
    eigs_true = np.array([0.95, 0.8])
    A = np.diag(eigs_true)
    x0 = np.array([1.0, 0.5])
    snapshots = [x0]
    for _ in range(20):
        snapshots.append(A @ snapshots[-1])
    X = np.column_stack(snapshots[:-1])
    Y = np.column_stack(snapshots[1:])

    eigs, _, _ = dmd_pair(X, Y, rank=2, center=False)
    eigs = np.sort(np.real_if_close(eigs))
    assert np.allclose(eigs, np.sort(eigs_true), atol=1e-10)
