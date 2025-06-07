import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_loader import load_subset
from clustering import cluster_data
from visualization import analyze_and_visualize
import logging

# Logging beállítása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Fő program.
    
    A program a következő lépéseket végzi:
    1. Adatok betöltése a data_loader segítségével
    2. Adatok előfeldolgozása
    3. Klaszterezés
    4. Eredmények elemzése és vizualizálása
    """
    # Adatok betöltése a data_loader segítségével
    data_subset = load_subset()
    
    # Adatok előfeldolgozása
    le = LabelEncoder()
    legal_areas = le.fit_transform(data_subset['JogTerulet'])
    legal_area_names = list(le.classes_)
    
    # Embeddingek átalakítása 2D numpy tömbbé
    embeddings = np.stack(data_subset['Embedding'].values)
    
    # Klaszterezés
    kmeans_labels, pca_data, optimal_k = cluster_data(embeddings, debug=True)
    
    # Eredmények elemzése és vizualizálása
    additional_metrics = analyze_and_visualize(
        pca_data, kmeans_labels, legal_areas, debug=True, legal_area_names=legal_area_names
    )
    
    # Eredmények kiírása
    logging.info(f"Optimális klaszterek száma: {optimal_k}")
    logging.info("\nKlaszterezési metrikák:")
    for metric_name, value in additional_metrics.items():
        logging.info(f"{metric_name}: {value:.3f}")

if __name__ == "__main__":
    main()