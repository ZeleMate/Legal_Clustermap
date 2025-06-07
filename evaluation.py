from sklearn.metrics import (
    silhouette_score, 
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_score,
    completeness_score,
    mutual_info_score,
    fowlkes_mallows_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def calculate_metrics(pca_data, kmeans_labels, le, debug=False):
    """
    Különböző klaszterezési metrikák kiszámítása.
    
    Paraméterek:
        pca_data (numpy.ndarray): A PCA transzformált adatok
        kmeans_labels (numpy.ndarray): A klaszterezés eredménye
        le (numpy.ndarray): A jogterületek LabelEncoder által kódolt értékei
        debug (bool): Debug információk kiírása
        
    Visszatérési érték:
        dict: A különböző metrikák értékei
    """
    metrics = {
        'silhouette': silhouette_score(pca_data, kmeans_labels),
        'calinski_harabasz': calinski_harabasz_score(pca_data, kmeans_labels),
        'davies_bouldin': davies_bouldin_score(pca_data, kmeans_labels),
        'homogeneity': homogeneity_score(le, kmeans_labels),
        'completeness': completeness_score(le, kmeans_labels),
        'mutual_info': mutual_info_score(le, kmeans_labels),
        'fowlkes_mallows': fowlkes_mallows_score(le, kmeans_labels),
        'adjusted_rand': adjusted_rand_score(le, kmeans_labels)
    }
    
    if debug:
        print("\n=== Klaszterezési metrikák ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")
    
    return metrics

def calculate_statistical_weights(metrics, debug=False):
    """
    Statisztikai alapú súlyok kiszámítása a metrikákhoz.
    
    A súlyok számítása több lépésben történik:
    1. Metrikák normalizálása
    2. Korrelációs mátrix számítása
    3. PCA alkalmazása a korrelációs mátrixon
    4. Súlyok számítása a fő komponensek alapján
    
    Paraméterek:
        metrics (dict): A különböző metrikák értékei
        debug (bool): Debug információk kiírása
        
    Visszatérési érték:
        dict: A számított súlyok
    """
    # Metrikák DataFrame-be konvertálása
    metrics_df = pd.DataFrame([metrics])
    
    # Normalizálás
    scaler = StandardScaler()
    metrics_normalized = scaler.fit_transform(metrics_df)
    
    # Korrelációs mátrix számítása
    corr_matrix = np.corrcoef(metrics_normalized.T)
    
    # NaN értékek kezelése
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Tökéletes korrelációk kezelése
    np.fill_diagonal(corr_matrix, 1.0)
    
    # PCA alkalmazása
    pca = PCA()
    pca.fit(corr_matrix)
    
    # Súlyok számítása a fő komponensek alapján
    # A súlyok arányosak a komponensek varianciájával
    weights = pca.explained_variance_ratio_
    
    # Ha minden súly 0 lenne, akkor egyenlő súlyozást alkalmazunk
    if np.all(weights == 0):
        weights = np.ones(len(metrics)) / len(metrics)
    else:
        # Súlyok normalizálása, hogy összegük 1 legyen
        weights = weights / weights.sum()
    
    # Súlyok hozzárendelése a metrikákhoz
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
    """
    Összetett pontszám kiszámítása a különböző metrikákból.
    
    A pontszám a következőképpen számolódik:
    1. Minden metrika normalizálva van [0,1] intervallumra
    2. A davies_bouldin score-t invertáljuk (1 - score)
    3. Statisztikai alapú súlyozott átlagot számolunk
    
    Paraméterek:
        metrics (dict): A különböző metrikák értékei
        debug (bool): Debug információk kiírása
        
    Visszatérési érték:
        float: Az összetett pontszám
    """
    # Davies-Bouldin score invertálása
    metrics['davies_bouldin'] = 1 - metrics['davies_bouldin']
    
    # Statisztikai súlyok számítása
    weights = calculate_statistical_weights(metrics, debug)
    
    # Összetett pontszám számítása
    composite_score = sum(metrics[metric] * weight for metric, weight in weights.items())
    
    return composite_score

def evaluate_clusters(data, kmeans_labels, pca_data, n_clusters=9, debug=False):
    """
    A klaszterezés eredményének értékelése és elemzése.
    
    A függvény a következő értékelési módszereket alkalmazza:
    1. Több különböző klaszterezési metrika
    2. Statisztikai alapú összetett pontszám számítása
    3. Részletes statisztikát készít a klaszterek eloszlásáról
    
    Paraméterek:
        data (pandas.DataFrame): Az eredeti adatok, tartalmazza a JogTerulet oszlopot
        kmeans_labels (numpy.ndarray): A klaszterezés eredménye
        pca_data (numpy.ndarray): A 2 dimenziós PCA transzformált adatok
        n_clusters (int): A klaszterek száma
        debug (bool): Debug információk kiírása
        
    Visszatérési érték:
        tuple: (metrics, composite_score, data_subset, le)
            - metrics: A különböző metrikák értékei
            - composite_score: Az összetett pontszám
            - data_subset: Az eredeti adatok a klaszter címkékkel kiegészítve
            - le: A jogterületek LabelEncoder által kódolt értékei
    """
    # Debug információk
    if debug:
        print("\n=== Értékelés Debug információk ===")
        print(f"Egyedi JogTerulet értékek: {data['JogTerulet'].unique()}")
        print(f"Egyedi klaszter címkék: {np.unique(kmeans_labels)}")
    
    # ARI: összevetés a címkékkel
    data_subset = data.copy()
    data_subset["cluster"] = kmeans_labels

    le = LabelEncoder().fit_transform(data_subset["JogTerulet"])
    
    if debug:
        print(f"\nJogTerulet és klaszter eloszlás:")
        distribution = data_subset[["JogTerulet", "cluster"]].groupby(["cluster", "JogTerulet"]).size()
        print(distribution)
        
        print("\nKlaszterenkénti eloszlás:")
        for cluster in range(n_clusters):
            cluster_data = data_subset[data_subset["cluster"] == cluster]
            print(f"\nKlaszter {cluster}:")
            print(cluster_data["JogTerulet"].value_counts())

    # Metrikák kiszámítása
    metrics = calculate_metrics(pca_data, kmeans_labels, le, debug)
    
    # Összetett pontszám számítása
    composite_score = calculate_composite_score(metrics, debug)
    
    if debug:
        print(f"\nÖsszetett pontszám: {composite_score:.3f}")

    return metrics, composite_score, data_subset, le