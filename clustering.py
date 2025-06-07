import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from metrics import find_optimal_clusters

def cluster_data(embedded_data, debug=False):
    """
    A beágyazott adatok klaszterezése és dimenziócsökkentése.
    
    A függvény a következő lépéseket végzi:
    1. PCA (Principal Component Analysis) segítségével 2 dimenzióra csökkenti az adatokat
    2. Automatikusan meghatározza az optimális klaszterek számát
    3. KMeans klaszterezést alkalmaz a csökkentett dimenziójú adatokon
    
    Paraméterek:
        embedded_data (numpy.ndarray): A beágyazott szövegek vektorai
        debug (bool): Ha True, akkor részletes debug információkat ír ki
        
    Visszatérési érték:
        tuple: (kmeans_labels, pca_data, optimal_k)
            - kmeans_labels: A klaszterezés eredménye, minden ponthoz tartozó klaszter címke
            - pca_data: A 2 dimenziós PCA transzformált adatok
            - optimal_k: Az optimális klaszterek száma
    """
    if debug:
        print("=== Debug információk ===")
        print("1. Adatok típusa és alakja:")
        print(f"embedded_data típusa: {type(embedded_data)}")
        print(f"embedded_data alakja: {embedded_data.shape}")
        print(f"embedded_data első néhány eleme:\n{embedded_data[:2]}")
        
        print("\n2. PCA előtti információk:")
        print(f"embedded_data dimenziói: {embedded_data.ndim}")
        print(f"embedded_data dtype: {embedded_data.dtype}")
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(embedded_data)
    
    if debug:
        print("\n3. PCA utáni információk:")
        print(f"pca_data típusa: {type(pca_data)}")
        print(f"pca_data alakja: {pca_data.shape}")
        print(f"pca_data első néhány eleme:\n{pca_data[:2]}")
    
    # Optimális klaszterek számának meghatározása
    optimal_k, _ = find_optimal_clusters(pca_data)
    
    # Klaszterezés az optimális klaszterek számával
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(pca_data)
    
    if debug:
        print("\n4. KMeans információk:")
        print(f"Optimalizált klaszterek száma: {optimal_k}")
        print(f"kmeans_labels típusa: {type(kmeans.labels_)}")
        print(f"kmeans_labels alakja: {kmeans.labels_.shape}")
        print(f"kmeans_labels első néhány eleme: {kmeans.labels_[:5]}")
    
    return kmeans.labels_, pca_data, optimal_k