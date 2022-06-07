import pandas as pd
from matplotlib import pyplot
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

FRAUD = 0
NORMAL = 1

df = pd.read_csv("creditcard_dataset_2022.csv")
df.V1.fillna(df.V1.mean(), inplace=True)


def kmeans_analysis(data, flip_cluster_index=True):
    kmeans = KMeans(n_clusters=2, random_state=212).fit(data)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data.index.values
    cluster_map['cluster'] = kmeans.labels_
    merged = pd.merge(data, cluster_map, left_index=True, right_on='data_index')
    if flip_cluster_index:
        merged['cluster'] = merged.cluster.apply(lambda x: 1 if x == 0 else 0)
    merged.rename(columns={'cluster': 'normal_prediction'}, inplace=True)
    merged.drop('data_index', inplace=True, axis=1)
    return merged


def get_true_frauds(kmeans_output):
    true_frauds = kmeans_output[(kmeans_output.normal_prediction == FRAUD) & (df.Normal == FRAUD)]
    print('True frauds captured: ', true_frauds.shape)
    return true_frauds


def get_false_normals(kmeans_output):
    false_normal = kmeans_output[(kmeans_output.normal_prediction == NORMAL) & (df.Normal == FRAUD)]
    print('False normal captured: ', false_normal.shape)
    return false_normal


def get_true_normals(kmeans_output):
    true_normal = kmeans_output[(kmeans_output.normal_prediction == NORMAL) & (df.Normal == NORMAL)]
    print('True normal captured: ', true_normal.shape)
    return true_normal


# Run kmeans on all columns except Amount and Target
data1 = df.drop(["Amount", "Normal"], axis=1)
data1_kmeans_results = kmeans_analysis(data1)
true_frauds_data1 = get_true_frauds(data1_kmeans_results)
false_frauds_data1 = get_false_normals(data1_kmeans_results)
true_normal_data1 = get_true_normals(data1_kmeans_results)


