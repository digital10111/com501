from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


def dist(X, centroid):
    return np.linalg.norm(X - centroid, ord=2, axis=1)


def get_nearest_centroid(X, centroids, num_centroids_chosen):
    min_dists = np.full((X.shape[0], num_centroids_chosen), np.inf)
    for i in range(num_centroids_chosen):
        min_dists[:, i] = dist(X, centroids[i])
    return np.argmin(min_dists, axis=1), np.min(min_dists, axis=1)


def kmeans_mb_plus_plus():
    centres = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = 2
    X = np.array([[5, 5], [4, 5], [4, 4], [1, 1], [1, 2], [2, 1]])

    K = 2
    colors = {0: 'green', 1: 'orange', 2: 'red', 3: 'yellow', 4: 'blue', 5: 'purple', 6: 'cyan', 7: 'pink', 8: 'brown', 9: 'black'}
    centroids = np.array([None] * K)

    num_centroids_chosen = 0
    centroids[num_centroids_chosen] = np.array([5, 5])
    num_centroids_chosen += 1

    moving_sums = np.zeros((K,))
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c='grey')

    while True:

        if num_centroids_chosen == K:
            break

        nearest_centroids, distances = get_nearest_centroid(X, centroids, num_centroids_chosen)
        distances_squared = np.square(distances)

        probabilites = distances_squared/distances_squared.sum()

        centroid_index = np.random.choice(np.arange(X.shape[0]), p=probabilites)

        centroids[num_centroids_chosen] = X[centroid_index]
        num_centroids_chosen += 1

        epochs = 0

        nearest_centroids, distances = get_nearest_centroid(X, centroids, num_centroids_chosen)

        local_cluster_point_sums = X[nearest_centroids == 0].sum(axis=0)
        print(local_cluster_point_sums/len(X[nearest_centroids == 0]))

        local_cluster_point_sums = X[nearest_centroids == 1].sum(axis=0)
        print(local_cluster_point_sums / len(X[nearest_centroids == 1]))

        l = 1



    plt.show()
    k = 1


kmeans_mb_plus_plus()


