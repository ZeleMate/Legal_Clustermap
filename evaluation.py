from sklearn.metrics import silhouette_score, adjusted_rand_score
from data_loader import load_data
from clustering import cluster_data
from visualization import visualize_clusters
from sklearn.preprocessing import LabelEncoder
import numpy as np

def evaluate_clusters(data, kmeans_labels, pca_data, le):
    # Debug információk
    print("\n=== Értékelés Debug információk ===")
    print(f"Egyedi JogTerulet értékek: {data['JogTerulet'].unique()}")
    print(f"Egyedi klaszter címkék: {np.unique(kmeans_labels)}")
    
    # ARI: összevetés a címkékkel
    data_subset = data.copy()  # Most már az összes sort használjuk
    data_subset["cluster"] = kmeans_labels

    le = LabelEncoder().fit_transform(data_subset["JogTerulet"])
    
    print(f"\nJogTerulet és klaszter eloszlás:")
    distribution = data_subset[["JogTerulet", "cluster"]].groupby(["cluster", "JogTerulet"]).size()
    print(distribution)
    
    print("\nKlaszterenkénti eloszlás:")
    for cluster in range(9):  # 9 klaszter
        cluster_data = data_subset[data_subset["cluster"] == cluster]
        print(f"\nKlaszter {cluster}:")
        print(cluster_data["JogTerulet"].value_counts())

    ari = adjusted_rand_score(le, data_subset["cluster"])
    print(f"\nAdjusted Rand Index: {ari:.3f}")
    # Silhouette score: belső koherencia
    silhouette = silhouette_score(pca_data, kmeans_labels)
    print(f"Silhouette Score: {silhouette:.3f}")

    return ari, silhouette, data_subset, le