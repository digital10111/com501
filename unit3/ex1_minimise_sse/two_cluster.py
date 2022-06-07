"""
Two Clusters case with a normal distribution
This time we need to be careful in terms of exact equality as the centroids search space is uniformed while the input space is normally distributed.
So the centroids will be close to the mean but not exactly the same.
"""

import matplotlib.pyplot as plt
import numpy as np

n = 301
a = 5

mu1, sigma1 = 1.5, 1.2
mu2, sigma2 = 10.0, 2.0

cluster1 = np.random.normal(mu1, sigma1, n)
cluster2 = np.random.normal(mu2, sigma2, n)

m1 = np.linspace(cluster1.min(), cluster1.max(), n*10)
m2 = np.linspace(cluster2.min(), cluster2.max(), n*10)

SSE1 = np.zeros(m1.shape)
SSE2 = np.zeros(m2.shape)

for i, mi in enumerate(m1):
    SSE1[i] = ((cluster1-mi)**2).sum()

for i, mi in enumerate(m2):
    SSE2[i] = ((cluster2-mi)**2).sum()


plt.plot(m1, SSE1, '-r')
plt.plot(m2, SSE2, '-b')
plt.grid()
plt.show()

print('----Cluster 1---------')
print(SSE1.min())
print(SSE1.argmin())
print(m1[SSE1.argmin()])
print(cluster1.mean())

print('----Cluster 2---------')
print(SSE2.min())
print(SSE2.argmin())
print(m2[SSE2.argmin()])
print(cluster2.mean())
