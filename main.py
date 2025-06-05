import pandas as pd
import os
import tqdm
from sentence_transformers import SentenceTransformer
from data_loader import load_data
from clustering import cluster_data

embedded_data, data = load_data(load_data)

kmeans_labels, pca_data = cluster_data(load_data)
