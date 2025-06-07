import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    mutual_info_score,
    fowlkes_mallows_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)

def calculate_additional_metrics(labels, legal_areas):
    """
    További metrikák számítása a klaszterezés értékeléséhez.
    
    Paraméterek:
        labels (numpy.ndarray): A klaszterezés eredménye
        legal_areas (numpy.ndarray): A jogi területek
        
    Visszatérési érték:
        dict: A számított metrikák
    """
    return {
        'homogeneity': homogeneity_score(legal_areas, labels),
        'completeness': completeness_score(legal_areas, labels),
        'mutual_info': mutual_info_score(legal_areas, labels),
        'fowlkes_mallows': fowlkes_mallows_score(legal_areas, labels),
        'adjusted_rand': adjusted_rand_score(legal_areas, labels),
        'adjusted_mutual_info': adjusted_mutual_info_score(legal_areas, labels)
    }

def plot_cluster_distribution(labels, legal_areas, legal_area_names=None, title="Klaszterek eloszlása a jogi területek szerint"):
    """
    Klaszterek eloszlásának ábrázolása a jogi területek szerint.
    
    Paraméterek:
        labels (numpy.ndarray): A klaszterezés eredménye
        legal_areas (numpy.ndarray): A jogi területek (kódolt vagy név)
        legal_area_names (dict vagy list): A kódolt értékekhez tartozó jogterület-nevek
        title (str): A diagram címe
    """
    # Jogterület nevek hozzárendelése
    if legal_area_names is None:
        unique_areas = np.unique(legal_areas)
        legal_area_names = {k: str(k) for k in unique_areas}
    elif isinstance(legal_area_names, list):
        legal_area_names = {i: name for i, name in enumerate(legal_area_names)}
    
    cluster_area_counts = {}
    for cluster in np.unique(labels):
        cluster_mask = labels == cluster
        area_counts = {}
        for area in np.unique(legal_areas):
            area_mask = legal_areas == area
            count = np.sum(cluster_mask & area_mask)
            if count > 0:
                area_counts[area] = count
        cluster_area_counts[cluster] = area_counts
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(cluster_area_counts))
    width = 0.8 / len(np.unique(legal_areas))
    
    for i, area in enumerate(np.unique(legal_areas)):
        plt.bar(x + i * width, [counts.get(area, 0) for counts in cluster_area_counts.values()],
                width, label=legal_area_names.get(area, str(area)))
    
    plt.xlabel('Klaszter')
    plt.ylabel('Darabszám')
    plt.title(title)
    plt.xticks(x + width * (len(np.unique(legal_areas)) - 1) / 2,
               [f'Klaszter {i}' for i in range(len(cluster_area_counts))])
    plt.legend(title="Jogi terület")
    plt.tight_layout()
    plt.show()

def plot_metrics(metrics, title="Klaszterezési metrikák"):
    """
    Klaszterezési metrikák ábrázolása.
    
    Paraméterek:
        metrics (dict): A klaszterezési metrikák
        title (str): A diagram címe
    """
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    plt.bar(x, list(metrics.values()))
    plt.xticks(x, list(metrics.keys()), rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_tsne(pca_data, labels, legal_areas, title="t-SNE ábrázolás a klaszterekről"):
    """
    t-SNE ábrázolás a klaszterekről.
    
    Paraméterek:
        pca_data (numpy.ndarray): A PCA transzformált adatok
        labels (numpy.ndarray): A klaszterezés eredménye
        legal_areas (numpy.ndarray): A jogi területek
        title (str): A diagram címe
    """
    # t-SNE alkalmazása
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(pca_data)
    
    # Ábrázolás
    plt.figure(figsize=(15, 8))
    
    # Klaszterek ábrázolása
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter, label='Klaszter')
    plt.title('Klaszterek')
    
    # Jogi területek ábrázolása
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=legal_areas, cmap='viridis')
    plt.colorbar(scatter, label='Jogi terület')
    plt.title('Jogi területek')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_and_visualize(pca_data, labels, legal_areas, debug=False, legal_area_names=None):
    """
    A klaszterezés eredményeinek elemzése és vizualizálása.
    
    Paraméterek:
        pca_data (numpy.ndarray): A PCA transzformált adatok
        labels (numpy.ndarray): A klaszterezés eredménye
        legal_areas (numpy.ndarray): A jogi területek
        debug (bool): Debug információk kiírása
        legal_area_names (dict vagy list): A kódolt értékekhez tartozó jogterület-nevek
    """
    if debug:
        print("\n=== Vizualizáció és elemzés ===")
        print("1. Adatok típusa és alakja:")
        print(f"pca_data típusa: {type(pca_data)}")
        print(f"pca_data alakja: {pca_data.shape}")
        print(f"labels típusa: {type(labels)}")
        print(f"labels alakja: {labels.shape}")
        print(f"legal_areas típusa: {type(legal_areas)}")
        print(f"legal_areas alakja: {legal_areas.shape}")
    
    # További metrikák számítása
    additional_metrics = calculate_additional_metrics(labels, legal_areas)
    
    if debug:
        print("\n2. További metrikák:")
        for metric_name, value in additional_metrics.items():
            print(f"{metric_name}: {value:.3f}")
    
    # Klaszterek eloszlása
    plot_cluster_distribution(labels, legal_areas, legal_area_names=legal_area_names)
    
    # Metrikák ábrázolása
    plot_metrics(additional_metrics)
    
    # t-SNE ábrázolás
    plot_tsne(pca_data, labels, legal_areas)
    
    return additional_metrics