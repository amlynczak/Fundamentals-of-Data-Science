from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import comb
from itertools import combinations
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram

#HIERARCHICAL CLUSTERING
print("hierarchical")

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

data = np.loadtxt("seeds.csv", delimiter=",")
#print(data)

data_clusters = data[:, :-1]

data_clusters = AgglomerativeClustering(3, compute_distances = True).fit(data)

print(rand_score(data[:,-1], data_clusters.labels_))
print(adjusted_rand_score(data[:,-1], data_clusters.labels_))

plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(data_clusters, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

print("--------")
#PARTITION CLUSTERING
print("partition")

from sklearn.cluster import KMeans

#data_kmeans = data[:, :-1]
data_kmeans = KMeans(n_clusters = 3, random_state = 0, n_init = "auto").fit(data)

print(rand_score(data[:,-1], data_kmeans.labels_))
print(adjusted_rand_score(data[:,-1], data_kmeans.labels_))
print("--------")
#REDUCED
print("reduced")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_red = pca.fit_transform(data)

#data_kmeans_red = data_red[:, :-1]
data_kmeans_red = KMeans(n_clusters = 3, random_state = 0, n_init = "auto").fit(data_red)

print(rand_score(data[:,-1], data_kmeans_red.labels_))
print(adjusted_rand_score(data[:,-1], data_kmeans_red.labels_))
