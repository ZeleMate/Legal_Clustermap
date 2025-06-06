from sklearn.metrics import silhouette_score, adjusted_rand_score
from data_loader import load_data
from clustering import cluster_data

def evaluate_clusters(data, kmeans_labels, pca_data):

    # ARI: összevetés a címkékkel
    data_subset = data.iloc[:100].copy()
    data_subset["cluster"] = kmeans_labels

    ari = adjusted_rand_score(data_subset["JogTerulet"], data_subset["cluster"])
    print(f"Adjusted Rand Index: {ari:.3f}")
    # Silhouette score: belső koherencia
    silhouette = silhouette_score(pca_data, kmeans_labels)
    print(f"Silhouette Score: {silhouette:.3f}")

    print(data_subset[["JogTerulet", "cluster"]].groupby(["cluster", "JogTerulet"]).size())

    return ari, silhouette, data_subset