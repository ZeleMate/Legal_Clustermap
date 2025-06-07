from data_loader import load_data, data_path
from clustering import cluster_data
from evaluation import evaluate_clusters
from visualization import visualize_clusters

def main(data, embedded_data, kmeans_labels, pca_data):
    ari, silhouette, data_subset, le = evaluate_clusters(data, kmeans_labels, pca_data, None)
    visualize_clusters(data_subset, kmeans_labels, pca_data, le)

    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Silhouette Score: {silhouette:.3f}")

    print(data_subset[["JogTerulet", "cluster"]].groupby(["cluster", "JogTerulet"]).size())

if __name__ == "__main__":
    embedded_data, data = load_data(data_path)
    kmeans_labels, pca_data = cluster_data(data, embedded_data)
    main(data, embedded_data, kmeans_labels, pca_data)