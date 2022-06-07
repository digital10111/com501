"""
Two clusters with two attributes
This time we will deal with two cluster and our datasets have two attributes x1 and x2
"""

import matplotlib.pyplot as plt
import numpy as np


n = 31 # number of points must be odd otherwise the samples on x will miss the minimum
a = 5

mu1, cov1 = [1.5, 5], [[1, 3], [2, 10]]
mu2, cov2 = [10.5, 5], [[1, 0], [0, 10]]


cluster1 = np.random.multivariate_normal(mu1, cov1, n).T
cluster2 = np.random.multivariate_normal(mu2, cov2, n).T

plt.plot(cluster1[0], cluster1[1], '.')
plt.plot(cluster2[0], cluster2[1], '.')

m10 = np.linspace(cluster1[0].min(), cluster1[0].max(), n*10)
m11 = np.linspace(cluster1[1].min(), cluster1[1].max(), n*10)

m20 = np.linspace(cluster2[0].min(), cluster2[0].max(), n*10)
m21 = np.linspace(cluster2[1].min(), cluster2[1].max(), n*10)


SSE1 = np.zeros((len(m10), len(m11)))
SSE2 = np.zeros((len(m20), len(m21)))


for i, mi in enumerate(m10):
    for j, mj in enumerate(m11):
        m1 = np.c_[mi, mj].T
        SSE1[i, j] = ((mi-cluster1[0])**2 + (mj-cluster1[1])**2).sum()


for i, mi in enumerate(m20):
    for j, mj in enumerate(m21):
        m2 = np.c_[mi, mj].T
        SSE2[i, j] = ((mi-cluster2[0])**2 + (mj-cluster2[1])**2).sum()


m_10, m_11 = np.meshgrid(m10, m11)
m_20, m_21 = np.meshgrid(m20, m21)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(m_10, m_11, SSE1)
surf = ax.plot_surface(m_20, m_21, SSE2)
plt.plot(cluster1[0], cluster1[1], '.')
plt.plot(cluster2[0], cluster2[1], '.')

print('----Cluster 1---------')
i, j = SSE1[0].argmin(), SSE1[1].argmin()
minSSE1 = np.c_[m_10[i, j], m_11[i, j]]
print(minSSE1)
print(cluster1.mean(1))

print('----Cluster 2---------')
#print(SSE2[0].min(), SSE2[1].min())
i,j = SSE2[0].argmin(), SSE2[1].argmin()
minSSE2 = np.c_[m_20[i,j], m_21[i,j]]
print(minSSE2)
print(cluster2.mean(1))

eps = 0.1
diff1 = cluster1.mean(1) - minSSE1
diff2 = cluster2.mean(1) - minSSE2

print(diff1.sum()/2 < eps and diff2.sum()/2 < eps)
plt.show()