import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from models.embedder import model
from data_loader import load_data
from clustering import cluster_data

def visualize_clusters(data_subset, kmeans_labels, pca_data, le):
    # Most már az összes sort használjuk
    colors = data_subset["JogTerulet"]

    plt.figure(figsize=(12, 5))

    # Klaszterek
    plt.subplot(1, 2, 1)
    plt.title("KMeans Clusters")
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap="tab10")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    # Valós jogterületek
    colors = le

    plt.subplot(1, 2, 2)
    plt.title("Actual JogTerulet")
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, cmap="tab10")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.tight_layout()
    plt.show()