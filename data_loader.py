import pandas as pd
from models.embedder import model
import numpy as np
data_path = '/Users/zelenyianszkimate/Documents/Szakdolgozat/processed_data/cleaned_data_for_embedding.csv'

def load_data(data_path):
    data = pd.read_csv(data_path)
    texts = data["text"].astype(str).tolist()[:100]
    embedded_data = np.array(model.encode(texts))
    return embedded_data, data