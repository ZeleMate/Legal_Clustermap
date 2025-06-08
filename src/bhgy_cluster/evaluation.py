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
import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        metrics = {}
        
        # Csak akkor számoljuk ki a metrikákat, ha van elég adat
        if len(np.unique(kmeans_labels)) > 1:
            try:
                metrics['silhouette'] = silhouette_score(pca_data, kmeans_labels)
            except:
                metrics['silhouette'] = 0.0
                
            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(pca_data, kmeans_labels)
            except:
                metrics['calinski_harabasz'] = 0.0
                
            try:
                metrics['davies_bouldin'] = davies_bouldin_score(pca_data, kmeans_labels)
            except:
                metrics['davies_bouldin'] = 1.0
        else:
            metrics['silhouette'] = 0.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = 1.0
    
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Ha nincs elég metrika, egyenlő súlyozást alkalmazunk
        if len(metrics) < 2:
            return {metric: 1.0/len(metrics) for metric in metrics}
        
        # Metrikák DataFrame-be konvertálása
        metrics_df = pd.DataFrame([metrics])
        
        # Normalizálás
        metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
        metrics_normalized = metrics_normalized.fillna(0)
        
        # Korrelációs mátrix számítása
        corr_matrix = metrics_normalized.corr()
        
        # NaN értékek kezelése
        corr_matrix = corr_matrix.fillna(0)
        
        # Tökéletes korrelációk kezelése
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        # Súlyok számítása a variancia alapján
        weights = np.var(metrics_normalized, axis=0)
        
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
            print(corr_matrix)
            print("\nSzámított súlyok:")
            for metric, weight in metric_weights.items():
                print(f"{metric}: {weight:.3f}")
        
        return metric_weights

def calculate_composite_score(metrics, debug=False):
    """
    Összetett pontszám kiszámítása a különböző metrikákból.
    
    Args:
        metrics (dict): A különböző metrikák értékei
        debug (bool): Debug mód bekapcsolása
        
    Returns:
        float: Az összetett pontszám (0 és 1 között)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Davies-Bouldin score invertálása
        metrics['davies_bouldin'] = 1 - metrics['davies_bouldin']
        
        # Statisztikai súlyok számítása
        weights = calculate_statistical_weights(metrics, debug)
        
        # Összetett pontszám számítása
        composite_score = sum(metrics[metric] * weight for metric, weight in weights.items())
        
        # Normalizálás 0 és 1 közé
        if composite_score < 0:
            composite_score = 0
        elif composite_score > 1:
            composite_score = 1
            
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
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