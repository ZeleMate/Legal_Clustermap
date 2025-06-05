from sklearn.metrics import silhouette_score, adjusted_rand_score
from data_loader import load_data
from clustering import cluster_data

def evaluate_clusters(data_subset, pca_data, kmeans):
# ARI: összevetés a címkékkel
    ari = adjusted_rand_score(data_subset["JogTerulet"], data_subset["cluster"])
    print(f"Adjusted Rand Index: {ari:.3f}")

    # Silhouette score: belső koherencia
    silhouette = silhouette_score(pca_data, kmeans.labels_)
    print(f"Silhouette Score: {silhouette:.3f}")

    return ari, silhouette