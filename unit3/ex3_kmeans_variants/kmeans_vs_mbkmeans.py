import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs


def generate_samples():
    batch_size = 45
    centres = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = len(centres)
    X, labels_true = make_blobs(n_samples=3000, centers=centres, cluster_std=0.7)
    return X, labels_true, batch_size, n_clusters


def kmeans(X):
    k_means = KMeans(init='k-means++', n_clusters=3, n_init=10, verbose=0)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0
    return k_means, t_batch


def mbKmeans(X, batch_size):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0
    return mbk, t_mini_batch


def cluster_plot(ax, X, points, center, col):
    ax.plot(X[points, 0], X[points, 1], 'o', markerfacecolor=col, marker='o', markersize=4)
    ax.plot(center[0], center[1], '+', markerfacecolor=col, markeredgecolor='k', markersize=16)


def set_fig(ax, title, time, intertia):
    ax.set_title(title)
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (time, intertia))


fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
# We want to have the same colours for the
X, labels_true, batch_size, n_clusters = generate_samples()

k_means, t_batch = kmeans(X)
k_centers = k_means.cluster_centers_

mbk, t_mini_batch = mbKmeans(X, batch_size)
order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
mbk_centers = mbk.cluster_centers_[order]

k_labels = pairwise_distances_argmin(X, k_centers)
mbk_labels = pairwise_distances_argmin(X, mbk_centers)

ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    clust_points = k_labels == k
    clust_center = k_centers[k]
    cluster_plot(ax, X, clust_points, clust_center, col)

ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    clust_points = mbk_labels == k
    clust_center = mbk_centers[k]
    cluster_plot(ax, X, clust_points, clust_center, col)

set_fig(ax, 'MiniBatchKMeans', t_mini_batch, intertia=mbk.inertia_)

different = (mbk_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    different += ((k_labels == k) != (mbk_labels == k))

identical = np.logical_not(different)
ax.plot(X[identical, 0], X[identical, 1], 'w', markerfacecolor='#bbbbbb', marker='.', markersize=12)
ax.plot(X[different, 0], X[different, 1], 'w', markerfacecolor='m', marker='.', markersize=12)

ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

plt.show()








