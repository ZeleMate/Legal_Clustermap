# BHGY_cluster
Clustering of court decisions.


BHGY_cluster/
│
├── __init__.py              # opcionális, a mappa modullá tétele
├── main.py                  # belépési pont
├── eval.py                  # csak kiértékelés
├── data_loader.py           # beolvasás + embed
├── clustering.py            # PCA + KMeans
├── evaluation.py            # ARI, Silhouette, vizualizáció
├── models/
│   ├── __init__.py
│   └── embedder.py          # SentenceTransformer
├── utils/
│   └── helpers.py           # kisebb segédfüggvények