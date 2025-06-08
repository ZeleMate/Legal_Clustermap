import pytest
import numpy as np
from metrics import calculate_metrics, find_optimal_clusters

@pytest.fixture
def sample_pca_data():
    """Teszt PCA adatok létrehozása."""
    np.random.seed(42)
    return np.random.rand(100, 2)  # 100 minta, 2 dimenziós PCA adatok

@pytest.fixture
def sample_labels():
    """Teszt klaszter címkék létrehozása."""
    np.random.seed(42)
    return np.random.randint(0, 3, size=100)  # 3 klaszter

def test_calculate_metrics_output(sample_pca_data, sample_labels):
    """Teszteli a metrikák számításának kimenetét."""
    metrics = calculate_metrics(sample_pca_data, sample_labels)
    
    assert isinstance(metrics, dict)
    assert 'silhouette' in metrics
    assert 'calinski_harabasz' in metrics
    assert 'davies_bouldin' in metrics
    
    # Ellenőrizzük a metrikák értékeinek érvényességét
    assert -1 <= metrics['silhouette'] <= 1
    assert metrics['calinski_harabasz'] >= 0
    assert metrics['davies_bouldin'] >= 0

def test_find_optimal_clusters_output(sample_pca_data):
    """Teszteli az optimális klaszterek számának meghatározását."""
    optimal_k, scores = find_optimal_clusters(sample_pca_data, max_clusters=5)
    
    assert isinstance(optimal_k, int)
    assert 2 <= optimal_k <= 5
    
    assert isinstance(scores, dict)
    assert 'silhouette_scores' in scores
    assert 'calinski_scores' in scores
    assert 'davies_scores' in scores
    assert 'composite_scores' in scores
    
    # Ellenőrizzük a pontszámok hosszát
    assert len(scores['silhouette_scores']) == 4  # 2-től 5-ig
    assert len(scores['calinski_scores']) == 4
    assert len(scores['davies_scores']) == 4
    assert len(scores['composite_scores']) == 4

def test_find_optimal_clusters_scores_range(sample_pca_data):
    """Teszteli a pontszámok tartományát és érvényességét."""
    _, scores = find_optimal_clusters(sample_pca_data, max_clusters=5)
    
    # Silhouette pontszámok: -1 és 1 között
    assert all(-1 <= score <= 1 for score in scores['silhouette_scores'])
    
    # Calinski-Harabasz: nem negatív
    assert all(score >= 0 for score in scores['calinski_scores'])
    
    # Davies-Bouldin: eredeti értékek >= 0, nem NaN, nem végtelen
    original_davies = [1 - s for s in scores['davies_scores']]
    assert all(np.isfinite(score) for score in original_davies)
    assert all(score >= 0 for score in original_davies)
    
    # Invertált Davies-Bouldin: nem NaN, nem végtelen
    assert all(np.isfinite(score) for score in scores['davies_scores'])
    
    # Composite score: 0 és 1 között
    assert all(0 <= score <= 1 for score in scores['composite_scores']) 