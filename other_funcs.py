# Import necessary libraries
import numpy as np
from numpy import linalg as la

def function_contour(func_eval, xmin=-20, xmax=20, ymin=-20, ymax=20, delta=0.1):
    x,y = np.arange(xmin, xmax, delta),np.arange(ymin, ymax, delta)
    X,Y = np.meshgrid(x,y)
    # Reshape the X,Y axes
    Xtemp = np.reshape(X,[np.prod(X.shape),1])
    Ytemp = np.reshape(Y,[np.prod(Y.shape),1])
    # Evaluate target and reshape
    Ztemp = func_eval(np.array([Xtemp,Ytemp]).T)
    Z = np.reshape(np.asmatrix(Ztemp),[X.shape[0],X.shape[1]])
    return X, Y, Z

def evaluate_mixture(x, rho, mu, sig):
    z = np.zeros(np.shape(x)[1])
    for i in range(np.shape(rho)[0]):
        z += rho[i]*mvn.pdf(x, mu[i], sig[i], allow_singular=True)
    return z

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False