import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from models.embedder import model

data = pd.read_csv('/Users/zelenyianszkimate/Documents/Szakdolgozat/processed_data/cleaned_data_for_embedding.csv')

embedded_data = data["text"].tolist()[:100]

embedded_data = model.encode(embedded_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(embedded_data)

kmeans = KMeans(n_clusters=5)
kmeans.fit(pca_data)

data_subset = data.iloc[:100].copy()
data_subset["cluster"] = kmeans.labels_

data_subset[["JogTerulet", "cluster"]].groupby(["cluster", "JogTerulet"]).size()



plt.figure(figsize=(12, 5))

# Klaszterek
plt.subplot(1, 2, 1)
plt.title("KMeans Clusters")
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data_subset["cluster"], cmap="tab10")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

# Valós jogterületek
le = LabelEncoder()
colors = le.fit_transform(data_subset["JogTerulet"])

plt.subplot(1, 2, 2)
plt.title("Actual JogTerulet")
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, cmap="tab10")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.tight_layout()
plt.show() 