import pytest
import numpy as np
from src.Legal_Clustermap.clustering import cluster_data

@pytest.fixture
def sample_embedded_data():
    """Tesztbeágyazott adatok létrehozása."""
    np.random.seed(42)
    return np.random.rand(100, 768)  # 100 minta, 768 dimenziós beágyazások

def test_cluster_data_output_types(sample_embedded_data):
    """Teszteli a klaszterezés kimenetének típusait."""
    kmeans_labels, pca_data, optimal_k = cluster_data(sample_embedded_data)
    
    assert isinstance(kmeans_labels, np.ndarray)
    assert isinstance(pca_data, np.ndarray)
    assert isinstance(optimal_k, int)
    
    # Ellenőrizzük a dimenziókat
    assert pca_data.shape[1] == 2  # PCA 2 dimenzióra csökkenti
    assert len(kmeans_labels) == len(sample_embedded_data)

def test_cluster_data_values(sample_embedded_data):
    """Teszteli a klaszterezés értékeinek érvényességét."""
    kmeans_labels, pca_data, optimal_k = cluster_data(sample_embedded_data)
    
    # Ellenőrizzük, hogy a klaszter címkék érvényesek
    assert optimal_k > 0
    assert len(np.unique(kmeans_labels)) == optimal_k
    assert all(label >= 0 and label < optimal_k for label in kmeans_labels)
    
    # Ellenőrizzük a PCA adatok érvényességét
    assert not np.isnan(pca_data).any()
    assert not np.isinf(pca_data).any()

def test_cluster_data_debug_mode(sample_embedded_data):
    """Teszteli a debug mód működését."""
    kmeans_labels, pca_data, optimal_k = cluster_data(sample_embedded_data, debug=True)
    
    # Debug módban is ugyanazokat a teszteket futtatjuk
    assert isinstance(kmeans_labels, np.ndarray)
    assert isinstance(pca_data, np.ndarray)
    assert isinstance(optimal_k, int)
    assert pca_data.shape[1] == 2 