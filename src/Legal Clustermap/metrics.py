import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def calculate_metrics(pca_data, labels):
    """
    Klaszterezési metrikák számítása.
    
    Paraméterek:
        pca_data (numpy.ndarray): A PCA transzformált adatok
        labels (numpy.ndarray): A klaszterezés eredménye
        
    Visszatérési érték:
        dict: A számított metrikák
    """
    return {
        'silhouette': silhouette_score(pca_data, labels),
        'calinski_harabasz': calinski_harabasz_score(pca_data, labels),
        'davies_bouldin': davies_bouldin_score(pca_data, labels)
    }

def find_optimal_clusters(pca_data, max_clusters=15):
    """
    Az optimális klaszterek számának meghatározása több metrika alapján.
    
    Paraméterek:
        pca_data (numpy.ndarray): A PCA transzformált adatok
        max_clusters (int): A maximális klaszterek száma
        
    Visszatérési érték:
        tuple: (optimal_k, scores)
            - optimal_k: Az optimális klaszterek száma
            - scores: A különböző klaszter számokhoz tartozó pontszámok
    """
    # Metrikák számítása
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    k_values = range(2, max_clusters + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pca_data)
        
        # Metrikák számítása
        metrics = calculate_metrics(pca_data, labels)
        silhouette_scores.append(metrics['silhouette'])
        calinski_scores.append(metrics['calinski_harabasz'])
        davies_scores.append(metrics['davies_bouldin'])
    
    # Davies-Bouldin score invertálása (kisebb érték jobb)
    davies_scores = [1 - score for score in davies_scores]
    
    # Összetett pontszám számítása
    composite_scores = []
    for i in range(len(k_values)):
        # Minden metrika normalizálása [0,1] intervallumra
        silhouette_norm = (silhouette_scores[i] - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores))
        calinski_norm = (calinski_scores[i] - min(calinski_scores)) / (max(calinski_scores) - min(calinski_scores))
        davies_norm = (davies_scores[i] - min(davies_scores)) / (max(davies_scores) - min(davies_scores))
        
        # Összetett pontszám: súlyozott átlag
        composite_score = (silhouette_norm + calinski_norm + davies_norm) / 3
        composite_scores.append(composite_score)
    
    # Az optimális k meghatározása
    optimal_k = k_values[np.argmax(composite_scores)]
    
    return optimal_k, {
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_scores': davies_scores,
        'composite_scores': composite_scores
    }

def calculate_statistical_weights(metrics, debug=False):
    metrics_df = pd.DataFrame([metrics])
    scaler = StandardScaler()
    metrics_normalized = scaler.fit_transform(metrics_df)
    corr_matrix = np.corrcoef(metrics_normalized.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    pca = PCA()
    pca.fit(corr_matrix)
    weights = pca.explained_variance_ratio_
    if np.all(weights == 0):
        weights = np.ones(len(metrics)) / len(metrics)
    else:
        weights = weights / weights.sum()
    metric_weights = dict(zip(metrics.keys(), weights))
    if debug:
        print("\n=== Statisztikai súlyok számítása ===")
        print("Korrelációs mátrix:")
        print(pd.DataFrame(corr_matrix, index=metrics.keys(), columns=metrics.keys()))
        print("\nFő komponensek varianciája:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {var:.3f}")
        print("\nSzámított súlyok:")
        for metric, weight in metric_weights.items():
            print(f"{metric}: {weight:.3f}")
    return metric_weights

def calculate_composite_score(metrics, debug=False):
    metrics['davies_bouldin'] = 1 - metrics['davies_bouldin']
    weights = calculate_statistical_weights(metrics, debug)
    composite_score = sum(metrics[metric] * weight for metric, weight in weights.items())
    return composite_score 