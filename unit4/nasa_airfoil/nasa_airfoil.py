import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


airfoil = pd.read_csv('../airfoil_self_noise.csv') #  we assume that you have this file in the same directory of this notebook!
D = airfoil.to_numpy()
D = np.asmatrix(D) # to


def frequency_feature():
    x = D[:, 0]
    t = D[:, 5]
    N = x.size
    x = ((x - x.min(0)) / (x.max(0) - x.min(0)))
    X = np.c_[np.ones(N), x]  # add dummy attribute
    w = np.ones((2, 1))  # initialise the weights
    y = X * w  # predict
    print(X[:5])


frequency_feature()


def shapeDF(DF, target, normalise=True):
    DFtarg = DF[[target]]
    DF = DF.drop([target], axis=1)
    x = DF.to_numpy()
    t = DFtarg.to_numpy()
    if normalise:
        x = ((x-x.min(0))/(x.max(0)-x.min(0)))
    return x, t


def LSQ(x, t):
    N = x.shape[0]
    M = x.shape[1]
    X, X[:, 1:] = np.ones((N, M+1)), x
    X = np.asmatrix(X)

    F = (X.T*X).I
    z = X.T*t
    w = F*z
    return w, X


def SSE(X, t, w):
    residuals = X*w-t
    sse = residuals.T*residuals
    return sse


def MSE(X, t, w):
    N = X.shape[0]
    MSE = (1.0/N)*SSE(X, t, w)
    return MSE


def RMSE(X, t, w):
    return np.sqrt(MSE(X, t, w))


def LSQ_Basis(x, t, fun_basis, k=1):
    phi = fun_basis(x, k)
    w, phi = LSQ(phi, t)
    rmse_ = RMSE(phi, t, w)
    return rmse_, w


def rbf_basis(X, k, vr=1):
    X = np.asarray(X)

    N = X.shape[0]
    M = X.shape[1]

    np.random.seed()
    eps = abs(np.random.rand(k, M))

    # means = X.mean(0)
    Mu = np.add(X.mean(0), eps)
    #sigma_inv = np.linalg.inv(np.cov(X, rowvar=False))
    sigma_inv = np.multiply(np.eye(M),1/X.var(0))
    phi = np.asarray(np.ones((N, k)))
    phi_1 = np.asarray(np.ones((N, k)))
    for b in range(k):
        Diff = (np.asmatrix(X) - Mu[b, :])
        DiffxS = Diff * sigma_inv
        phi_1[:,b] = np.exp(-0.5*np.sum(np.multiply(Diff,DiffxS),1).reshape(1503))

    for b in range(k):
        for i in range(N):
            diff = (X[i, :] - Mu[b, :])
            res = (diff.dot(sigma_inv.T)).dot(diff.T)
            phi[i, b] = np.exp(-0.5 * res)
    sub = (phi_1 - phi).sum()
    return phi


x,t = shapeDF(airfoil, target='Scaled_sound_pressure_level(decibels)')
i = 1
while i < 1000:
    k = np.random.randint(1, 100)  # number of Gaussian basis

    RMSE4, w = LSQ_Basis(x, t, rbf_basis, k)

    if i == 1 or RMSE4 < RMSE4_best:
        RMSE4_best, w_best = RMSE4, w
        print(RMSE4_best)
        print(w_best)
        print("**\n")

    i += 1

print('best weights have ', w_best.shape[0], ' components \nRoot Mean of Squared Errors = ',
      RMSE4_best, 'and  RMSE4 < RMSE3 is', RMSE4_best <= RMSE4)
