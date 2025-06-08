import pytest
import numpy as np
import pandas as pd
from src.bhgy_cluster.evaluation import evaluate_clusters
from src.bhgy_cluster.evaluation import (
    calculate_metrics,
    calculate_statistical_weights,
    calculate_composite_score
)

@pytest.fixture
def sample_data():
    """Tesztadatok létrehozása."""
    # Több adatpontot generálunk, hogy elkerüljük a statisztikai problémákat
    np.random.seed(42)
    n_samples = 100
    pca_data = np.random.rand(n_samples, 2)
    kmeans_labels = np.random.randint(0, 3, n_samples)
    data = pd.DataFrame({
        'text': [f'Teszt szöveg {i}' for i in range(n_samples)],
        'JogTerulet': np.random.choice(['Polgári', 'Büntető', 'Közigazgatási'], n_samples)
    })
    return pca_data, kmeans_labels, data

def test_calculate_metrics(sample_data):
    """Teszteli a metrikák számítását."""
    pca_data, kmeans_labels, _ = sample_data
    le = np.random.randint(0, 3, len(kmeans_labels))  # Teszt címkék
    
    metrics = calculate_metrics(pca_data, kmeans_labels, le)
    
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ['silhouette', 'calinski_harabasz', 'davies_bouldin'])
    assert all(isinstance(value, (int, float)) for value in metrics.values())

def test_calculate_statistical_weights(sample_data):
    """Teszteli a statisztikai súlyok számítását."""
    pca_data, kmeans_labels, _ = sample_data
    le = np.random.randint(0, 3, len(kmeans_labels))
    
    # Először számoljuk ki a metrikákat
    metrics = calculate_metrics(pca_data, kmeans_labels, le)
    
    # Majd számoljuk ki a súlyokat
    weights = calculate_statistical_weights(metrics)
    
    assert isinstance(weights, dict)
    assert len(weights) == len(metrics)
    assert all(isinstance(value, (int, float)) for value in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-10  # A súlyok összege 1

def test_calculate_composite_score(sample_data):
    """Teszteli az összetett pontszám számítását."""
    pca_data, kmeans_labels, _ = sample_data
    le = np.random.randint(0, 3, len(kmeans_labels))
    
    # Először számoljuk ki a metrikákat
    metrics = calculate_metrics(pca_data, kmeans_labels, le)
    
    # Majd számoljuk ki az összetett pontszámot
    score = calculate_composite_score(metrics)
    
    assert isinstance(score, (int, float))
    assert 0 <= score <= 1  # Az összetett pontszám 0 és 1 között van

def test_evaluate_clusters_output(sample_data):
    """Teszteli a klaszterezés értékelésének kimenetét."""
    pca_data, kmeans_labels, data = sample_data
    
    metrics, composite_score, data_subset, le = evaluate_clusters(
        data, kmeans_labels, pca_data, n_clusters=3
    )
    
    assert isinstance(metrics, dict)
    assert isinstance(composite_score, (int, float))
    assert isinstance(data_subset, pd.DataFrame)
    assert isinstance(le, np.ndarray)
    assert len(le) == len(data_subset)
    assert 'cluster' in data_subset.columns 