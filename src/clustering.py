from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_kmeans(X,min_k=2,max_k=10):
    result = []

    for k in range(min_k,max_k+1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            init='k-means++'
        )

        labels = model.fit_predict(X)

        result.append({
            "K":k,
            "WCSS":model.inertia_ ,
            "Silhouette" : silhouette_score(X,labels)

        })

    return pd.DataFrame(result)


def train_kmeans(X,n_clusters):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        init='k-means++'
    )

    labels = model.fit_predict(X)

    return model,labels


def evaluate_hierarchical(
    X,
    min_k=2,
    max_k=10
):

    results = []

    for k in range(min_k, max_k + 1):

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels
        )

        results.append({
            "K": k,
            "Silhouette": score
        })

    return pd.DataFrame(results)



def train_hierarchical(
        X,
        n_clusters
):
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )

    labels = model.fit_predict(X)

    return labels



def plot_dendrogram(
    X,
):
    linked = linkage(
        X,
        method="ward"
    )

    plt.figure(figsize=(15,6))

    dendrogram(
        linked,
        truncate_mode="level",
        p=5
    )

    plt.title(
        "Hierarchical Clustering Dendrogram"
    )

    plt.xlabel("Observations")
    plt.ylabel("Distance")

    plt.show()