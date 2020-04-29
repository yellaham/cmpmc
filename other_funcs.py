# Import necessary libraries
import numpy as np
from numpy import linalg as la
from scipy.stats import multivariate_normal as mvn


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
