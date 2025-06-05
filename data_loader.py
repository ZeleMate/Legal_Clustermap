import pandas as pd
from models.embedder import model

data_path = '/Users/zelenyianszkimate/Documents/Szakdolgozat/processed_data/cleaned_data_for_embedding.csv'

def load_data(data_path):
    data = pd.read_csv(data_path)
    embedded_data = data["text"].tolist()[:100]
    embedded_data = model.encode(embedded_data)
    return embedded_data, data