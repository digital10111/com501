import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generateC12(n=50, spread=2):
    mu1, cov1 = [5, 5], [[.5*spread, 0], [0, 0.5*spread]]
    mu2, cov2 = [9, 5], [[.5*spread, 0], [0, .5*spread]]

    N = n
    N1, N2 = N, N

    C1 = np.round(np.random.multivariate_normal(mu1, cov1, N1), 1)
    C2 = np.round(np.random.multivariate_normal(mu2, cov2, N2), 1)
    C12 = np.concatenate((C1, C2), 0)
    c1 = np.average(C1, 0)
    c2 = np.average(C2, 0)

    return C1, C2, c1, c2, C12


def plotC(C, colourshape='bo', c=[]):
    C = C.T
    plt.plot(C[0], C[1], colourshape, markersize=10, mfc='none')
    if len(c) != 0:
        plt.plot(c[0], c[1], '+', markersize=25)
    plt.xlim([0, 11])
    plt.ylim([0, 11])


def plotC12(C1, C2, c1, c2):
    plotC(C1, 'g^', c1)
    plotC(C2, 'yo', c2)


def getClusters(C12, c1, c2):
    dist1 = np.sqrt(np.sum((c1-C12)**2, 1))
    dist2 = np.sqrt(np.sum((c2-C12)**2, 1))
    min_dist = np.minimum(dist1, dist2)

    C1_ind = np.where(dist1 == min_dist)
    C2_ind = np.where(dist2 == min_dist)

    C1 = C12[C1_ind]
    C2 = C12[C2_ind]

    return C1, C2


def getCentroids(C1, C2):
    c1 = np.average(C1, 0)
    c2 = np.average(C2, 0)
    return c1, c2


def kmeans(C12, c1, c2):
    c1, c2 = c1, c2

    for i in range(10):
        C1, C2 = getClusters(C12, c1, c2)
        c1_, c2_ = getCentroids(C1, C2)

        if (c1_ == c1).all() and (c2_ == c2).all():
            break
        else:
            c1, c2 = c1_, c2_


def kmeansMB(C12, c1, c2):
    C1, C2 = getClusters(C12, c1, c2)
    mvs_c1 = 0
    mvs_c2 = 0
    c1, c2 = np.array(c1), np.array(c2)
    np.random.shuffle(C12)
    batches = np.split(C12, 5, axis=0)
    for i in range(10):
        for batch in batches:
            C1_, C2_ = getClusters(batch, c1, c2)
            c1 = (c1 * mvs_c1 + C1_.sum())/(mvs_c1 + len(C1_))
            c2 = (c2 * mvs_c2 + C2_.sum())/(mvs_c2 + len(C2_))
            mvs_c1 = (mvs_c1 + len(C1_))
            mvs_c2 = (mvs_c2 + len(C2_))
        C1, C2 = getClusters(C12, c1, c2)
        c1, c2 = getCentroids(C1, C2)
        mvs_c1 = len(C1)
        mvs_c2 = len(C2)
    C1, C2 = getClusters(C12, c1, c2)
    c1, c2 = getCentroids(C1, C2)
    return C1, C2, c1, c2

C1, C2, c1, c2, C12 = generateC12()
print(C12.shape)
#plotC12(C1, C2, c1+0.7, c2-0.7 )
# plotC(C12)


c1, c2 = [2, 2], [7, 2]
C1, C2, c1, c2 = kmeansMB(C12, c2, c1)
plotC12(C1, C2, c1, c2)
plt.show()
