import pytest
import numpy as np
import pandas as pd
from evaluation import (
    calculate_metrics,
    calculate_statistical_weights,
    calculate_composite_score,
    evaluate_clusters
)

@pytest.fixture
def sample_data():
    """Tesztadatok létrehozása."""
    return pd.DataFrame({
        'JogTerulet': ['Polgári', 'Büntető', 'Polgári', 'Büntető', 'Polgári']
    })

@pytest.fixture
def sample_pca_data():
    """Teszt PCA adatok létrehozása."""
    np.random.seed(42)
    return np.random.rand(5, 2)  # 5 minta, 2 dimenziós PCA adatok

@pytest.fixture
def sample_labels():
    """Teszt klaszter címkék létrehozása."""
    return np.array([0, 1, 0, 1, 0])

@pytest.fixture
def sample_metrics():
    """Teszt metrikák létrehozása."""
    return {
        'silhouette': 0.5,
        'calinski_harabasz': 0.7,
        'davies_bouldin': 0.3,
        'homogeneity': 0.6,
        'completeness': 0.8,
        'mutual_info': 0.4,
        'fowlkes_mallows': 0.9,
        'adjusted_rand': 0.75
    }

def test_calculate_metrics_output(sample_pca_data, sample_labels, sample_data):
    """Teszteli a metrikák számításának kimenetét."""
    le = pd.factorize(sample_data['JogTerulet'])[0]
    metrics = calculate_metrics(sample_pca_data, sample_labels, le)
    
    assert isinstance(metrics, dict)
    expected_metrics = [
        'silhouette', 'calinski_harabasz', 'davies_bouldin',
        'homogeneity', 'completeness', 'mutual_info',
        'fowlkes_mallows', 'adjusted_rand'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)

def test_calculate_statistical_weights(sample_metrics):
    """Teszteli a statisztikai súlyok számítását."""
    weights = calculate_statistical_weights(sample_metrics)
    
    assert isinstance(weights, dict)
    assert len(weights) == len(sample_metrics)
    
    # Ellenőrizzük, hogy a súlyok összege 1
    assert abs(sum(weights.values()) - 1.0) < 1e-10
    
    # Ellenőrizzük, hogy minden súly nem negatív
    assert all(weight >= 0 for weight in weights.values())

def test_calculate_composite_score(sample_metrics):
    """Teszteli az összetett pontszám számítását."""
    score = calculate_composite_score(sample_metrics)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_evaluate_clusters_output(sample_data, sample_pca_data, sample_labels):
    """Teszteli a klaszterezés értékelésének kimenetét."""
    metrics, composite_score, data_subset, le = evaluate_clusters(
        sample_data, sample_labels, sample_pca_data, n_clusters=2
    )
    
    assert isinstance(metrics, dict)
    assert isinstance(composite_score, float)
    assert isinstance(data_subset, pd.DataFrame)
    assert isinstance(le, np.ndarray)
    
    # Ellenőrizzük a data_subset tartalmát
    assert 'cluster' in data_subset.columns
    assert len(data_subset) == len(sample_data)
    
    # Ellenőrizzük a le hosszát
    assert len(le) == len(sample_data) 