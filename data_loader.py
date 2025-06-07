import pandas as pd
from models.embedder import model
import numpy as np
data_path = '/Users/zelenyianszkimate/Documents/Szakdolgozat/processed_data/cleaned_data_for_embedding.csv'

def load_data(data_path):
    data = pd.read_csv(data_path)
    print("\n=== Adatok betöltése Debug ===")
    print(f"Összes sor száma: {len(data)}")
    print("\nJogterületek eloszlása az egész adatbázisban:")
    print(data["JogTerulet"].value_counts())
    
    # Véletlenszerűen választunk 1000 sort
    sample_size = 1000
    sampled_data = data.sample(n=sample_size, random_state=42)  # random_state a reprodukálhatóság miatt
    
    texts = sampled_data["text"].astype(str).tolist()
    print(f"\nBetöltött szövegek száma: {len(texts)}")
    print("\nJogterületek eloszlása a véletlenszerűen választott sorokban:")
    print(sampled_data["JogTerulet"].value_counts())
    
    embedded_data = np.array(model.encode(texts))
    return embedded_data, sampled_data