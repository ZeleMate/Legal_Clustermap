import pandas as pd
from models.embedder import model
import numpy as np
import logging
import os
from dotenv import load_dotenv

# Környezeti változók betöltése
load_dotenv()

# Adatok elérési útja környezeti változóból vagy relatív alapértelmezéssel
DATA_PATH = os.getenv('DATA_PATH')

def load_data():
    """
    Adatok betöltése a CSV fájlból.
    
    Visszatérési érték:
        pandas.DataFrame: A betöltött adatok
    """
    data = pd.read_csv(DATA_PATH)
    logging.info(f"Összes sor száma: {len(data)}")
    logging.info("\nJogterületek eloszlása az egész adatbázisban:")
    logging.info(data["JogTerulet"].value_counts())
    return data

def load_subset(n_samples=1000, random_state=42):
    """
    Véletlenszerűen kiválasztott adatok betöltése és beágyazása.
    
    Paraméterek:
        n_samples (int): A kiválasztandó sorok száma
        random_state (int): A véletlenszerűség seed értéke
        
    Visszatérési érték:
        pandas.DataFrame: A kiválasztott és beágyazott adatok
    """
    # Adatok betöltése
    data = load_data()
    
    # Véletlenszerűen választunk n_samples sort
    sampled_data = data.sample(n=n_samples, random_state=random_state)
    
    # Szövegek beágyazása
    texts = sampled_data["text"].astype(str).tolist()
    embedded_data = np.array(model.encode(texts))
    
    # Beágyazott adatok hozzáadása
    sampled_data['Embedding'] = list(embedded_data)
    
    logging.info(f"\nBetöltött szövegek száma: {len(texts)}")
    logging.info("\nJogterületek eloszlása a véletlenszerűen választott sorokban:")
    logging.info(sampled_data["JogTerulet"].value_counts())
    
    return sampled_data