import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
import time


def vanilla_GD(X, t, eta=0.01, maxep=100, decay=0.98):
    start_time = time.time()

    N = len(t)
    w = np.zeros((2, 1))
    J = np.zeros((maxep))

    for ep in range(maxep):
        for n in range(N):
            w += (1.0 / N) * eta * X[n, :].T * (t[n] - w.T * X[n, :].T)
            J[ep] += (1.0 / (2 * N)) * (t[n] - w.T * X[n, :].T) ** 2

        eta *= decay

    tme = time.time() - start_time
    return w, J, tme


def vectrzd_GD(X, t, eta=0.001, maxep=100, decay=0.98):
    start_time = time.time()

    N = len(t)
    w = np.zeros((2, 1))
    J = np.zeros(maxep)

    for ep in range(maxep):
        w += (1.0 / N) * eta * X.T * (t - X * w)
        J[ep] += (1.0 / (2 * N)) * (t - X * w).T * (t - X * w)
        eta *= decay
    tme = time.time() - start_time
    return w, J, tme


def vectrzd_batch_SGD(X, t, eta=0.001, maxep=100, decay=0.98, b=25):
    start_time = time.time()

    N = len(t)
    w = np.zeros((2, 1))
    J = np.zeros(maxep)

    for ep in range(maxep):
        for tau in range(int(N/b)):
            batch_X = X[tau*b:(tau+1)*b]
            w += (1.0 / b) * eta * batch_X.T * (t - batch_X * w)
            J[ep] += (1.0 / (2 * N)) * (t - batch_X * w).T * (t - batch_X * w)
        eta *= decay
    tme = time.time() - start_time
    return w, J, tme


def outGD(J, w, tme, name='GD', ax=None, plotJ=True, printw=True):
    if plotJ:
        ax = ax or plt.gca()
        ax.plot(J[1:], '.b')
    if printw:
        print("y = {} + {} X1".format(w[0], w[1]), " {} took:    {} seconds ".format(name, tme))


def real_func(x1):
    return 2 + 5 * x1


def generate_data(N=100, noise=5):
    N = N
    noise = noise

    x1 = np.linspace(-10, 10, N)
    t = np.random.normal(real_func(x1), noise)

    plt.plot(x1, t, ".")
    plt.plot(x1, real_func(x1), 'r')

    return x1, t


def prepare_date(x1, t):
    x0 = np.ones(len(x1))
    X = np.c_[x0, x1]
    X = np.matrix(X)
    t = np.matrix(t).T
    return X, t


x1, t = generate_data()
X, t = prepare_date(x1, t)

#
maxep = 100
# eta = 0.01
decay = 0.98
#
# fig, (ax1, ax2) = plt.subplots(2)
#
# w, J, tme = vanilla_GD(X, t, eta, maxep, decay)
# outGD(J, w, tme, 'vallina_GD', ax1)
#
# w,J, tme = vectrzd_GD(X, t, eta, maxep, decay)
# outGD(J, w, tme, 'GD_vectrzd', ax2)


etasL = np.arange(0.01, 0.15, 0.005)
etasH = np.arange(0.16, 0.5, 0.05)

etas = np.concatenate((etasL, etasH))

Jv = np.zeros(len(etas))
Jz = np.zeros(len(etas))

for i, eta in enumerate(etas):
    w_v, J_v, _ = vanilla_GD(X, t, eta, maxep, decay)
    w_z, J_z, _ = vectrzd_GD(X, t, eta, maxep, decay)

    Jv[i] = np.sum(J_v[-10:-1]/10.0)
    Jz[i] = np.sum(J_z[-10:-1]/10.0)


fig, (ax1, ax2) = plt.subplots(2, 2)
fig.legend()

ax1[0].plot(etasL,Jv[:len(etasL)],'.r', label='Avg loss for low $\\eta$ on vanilla')
ax2[0].plot(etasL,Jz[:len(etasL)],'.b',  label='Avg loss for low $\\eta$ on vectorised')

ax1[1].plot(etasH,Jv[len(etasL):], '--r', label='Avg loss for high $\\eta$ on vanilla')
ax2[1].plot(etasH,Jz[len(etasL):],'--b', label='Avg loss for high $\\eta$ on vectorised')

fig.legend(ncol=2)
fig.tight_layout()

maxep = 40
eta = 0.005
decay = 0.98



plt.show()

