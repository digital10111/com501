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
    n_clusters = len(centres)
    X, labels_true = make_blobs(n_samples=3000, centers=centres, cluster_std=0.7)
    K = n_clusters
    colors = {0: 'green', 1: 'orange', 2: 'red', 3: 'yellow', 4: 'blue', 5: 'purple', 6: 'cyan', 7: 'pink', 8: 'brown', 9: 'black'}
    centroids = np.array([None] * K)

    num_centroids_chosen = 0
    centroids[num_centroids_chosen] = np.random.permutation(X)[:1].squeeze()
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
        for i in range(num_centroids_chosen):
            moving_sums[i] = np.count_nonzero(nearest_centroids == i)

        # fig, ax = plt.subplots()
        # ax.scatter(X[:, 0], X[:, 1], c='grey')
        for i in range(num_centroids_chosen):
            ax.scatter(centroids[i][0], centroids[i][1], c=colors[i])
        # plt.show()

        while epochs < 10:
            batches = np.split(X, 10, axis=0)
            for batch in batches:
                local_nearest_centroids, local_distances = get_nearest_centroid(batch, centroids, num_centroids_chosen)
                for i in range(num_centroids_chosen):
                    centroid_moving_average = centroids[i] * moving_sums[i]
                    local_cluster_point_sums = batch[local_nearest_centroids == i].sum(axis=0)
                    numer = centroid_moving_average + local_cluster_point_sums

                    moving_sum = moving_sums[i]
                    num_point_belonging_to_cluster = np.count_nonzero(local_nearest_centroids == i)
                    denom = moving_sum + num_point_belonging_to_cluster
                    if denom:
                        centroids[i] = numer/denom
                        moving_sums[i] = denom
                        k = 1
                    else:
                        print("what")
            epochs += 1
            nearest_centroids, distances = get_nearest_centroid(X, centroids, num_centroids_chosen)
            moving_sums = np.zeros((K,))
            for i in range(num_centroids_chosen):
                moving_sums[i] = np.count_nonzero(nearest_centroids == i)

    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c='grey')
    for i in range(num_centroids_chosen):
        ax.scatter(centroids[i][0], centroids[i][1], c=colors[i])
    plt.show()
    k = 1


kmeans_mb_plus_plus()


