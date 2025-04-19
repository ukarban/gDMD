import numpy as np
from numpy import linalg

def dmd_pair(X1, X2, r, dt):
    ## STEP 1: singular value decomposition (SVD)
    transpose = lambda x: np.transpose(x)
    hermitian = lambda x: np.conj(np.transpose(x))
    X1n = X1 - np.mean(X1,axis=1,keepdims=True)
    X2n = X2 - np.mean(X2,axis=1,keepdims=True)
    [m, n] = X1n.shape
    dmin = np.min((m,n))
    [U, Sdiag, V] = linalg.svd(X1n, full_matrices=False)
    S = np.zeros((m, n), dtype=Sdiag.dtype)
    S[:dmin, :dmin] = np.diag(Sdiag)
    V = hermitian(V)

    Ur = U[:, :r]
    Sr = S[:r, :r]
    Vr = V[:, :r]

    ## STEP 2: low-rank subspace matrix
    # (similarity transform, least-square fit matrix, low-rank subspace matrix)
    # Atilde = np.dot(np.dot(np.dot(Ur.transpose(), X2), Vr), linalg.inv(Sr))
    Atilde = np.dot(hermitian(Ur), np.dot(X2n, np.dot(Vr, linalg.inv(Sr))))
    # Atilde = hermitian(Ur) @ X2 @ Vr @ linalg.inv(Sr)

    ## STEP 3: eigen decomposition
    # W: eigen vectors
    # D: eigen values
    [m, n] = Atilde.shape
    [Ddiag, W] = linalg.eig(Atilde)
    isort = np.argsort(np.abs(Ddiag))[::-1]
    Ddiag = Ddiag[isort]
    W = W[:,isort]
    D = np.zeros((m, n), dtype=Ddiag.dtype)
    Dinv = np.zeros((m, n), dtype=Ddiag.dtype)
    D[:n, :n] = np.diag(Ddiag)
    Dinv[:n, :n] = np.diag(1/Ddiag)
    

    ## STEP 4: real space DMD mode
    Phi = np.dot(np.dot(X2n, np.dot(Vr, np.dot(linalg.inv(Sr), W))),Dinv)
    Phi_proj = np.dot(Ur, W)

    sgm = np.diag(D)
    omega = np.log(sgm)/dt
    return sgm, Phi, Phi_proj

def dmd_serial(X1,r):
    X1n = X1 - np.mean(X1,axis=1,keepdims=True)
    coeff,_,_,_= linalg.lstsq(X1n[:,:-1],X1n[:,-1],rcond=None)
    coeffMat = np.eye(X1n.shape[1]-1,k=-1)
    coeffMat[:,-1] = coeff
    sgm, Phi = linalg.eig(coeffMat)
    sgm = sgm[:r]
    Phi = X1n[:,:-1]@Phi[:,:r]

    return sgm, Phi
