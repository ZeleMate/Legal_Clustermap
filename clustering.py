import pandas as pd
import numpy as np
from data_loader import load_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def cluster_data(data, embedded_data):
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
    
    print("\n3. PCA utáni információk:")
    print(f"pca_data típusa: {type(pca_data)}")
    print(f"pca_data alakja: {pca_data.shape}")
    print(f"pca_data első néhány eleme:\n{pca_data[:2]}")
    
    kmeans = KMeans(n_clusters=9, random_state=42)
    kmeans.fit(pca_data)
    
    print("\n4. KMeans információk:")
    print(f"kmeans_labels típusa: {type(kmeans.labels_)}")
    print(f"kmeans_labels alakja: {kmeans.labels_.shape}")
    print(f"kmeans_labels első néhány eleme: {kmeans.labels_[:5]}")
    
    return kmeans.labels_, pca_data