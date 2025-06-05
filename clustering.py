from data_loader import load_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def cluster_data(data_path):
    embedded_data, data = load_data(data_path)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(embedded_data)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pca_data)

    return kmeans.labels_, pca_data