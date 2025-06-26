import pandas as pd
from bhgy_cluster.models.embedder import model
import numpy as np
import logging
import os
from dotenv import load_dotenv

# Környezeti változók betöltése
load_dotenv()

# Adatok elérési útja környezeti változóból vagy relatív alapértelmezéssel
DATA_PATH = os.getenv('DATA_PATH')

def get_test_data():
    """Tesztadatok létrehozása."""
    return pd.DataFrame({
        'text': ['Teszt szöveg 1', 'Teszt szöveg 2', 'Teszt szöveg 3'],
        'JogTerulet': ['Polgári', 'Büntető', 'Polgári']
    })

def load_data():
    """
    Adatok betöltése a CSV fájlból.
    
    Visszatérési érték:
        pandas.DataFrame: A betöltött adatok
    """
    try:
        if not DATA_PATH or not os.path.exists(DATA_PATH):
            logging.warning(f"A {DATA_PATH} fájl nem létezik, tesztadatok használata.")
            return get_test_data()
            
        data = pd.read_csv(DATA_PATH)
        logging.info(f"Összes sor száma: {len(data)}")
        logging.info("\nJogterületek eloszlása az egész adatbázisban:")
        logging.info(data["JogTerulet"].value_counts())
        return data
    except Exception as e:
        logging.error(f"Hiba történt az adatok betöltésekor: {str(e)}")
        return get_test_data()

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
    
    # Ha nincs elég adat, akkor minden sort használunk
    n_samples = min(n_samples, len(data))
    
    # Véletlenszerűen választunk n_samples sort
    sampled_data = data.sample(n=n_samples, random_state=random_state)
    
    try:
        # Szövegek beágyazása
        texts = sampled_data["text"].astype(str).tolist()
        embedded_data = np.array(model.encode(texts))
        
        # Beágyazott adatok hozzáadása
        sampled_data['Embedding'] = list(embedded_data)
        
        logging.info(f"\nBetöltött szövegek száma: {len(texts)}")
        logging.info("\nJogterületek eloszlása a véletlenszerűen választott sorokban:")
        logging.info(sampled_data["JogTerulet"].value_counts())
        
        return sampled_data
    except Exception as e:
        logging.error(f"Hiba történt a beágyazás során: {str(e)}")
        # Ha hiba történik a beágyazás során, visszaadjuk az eredeti adatokat
        return sampled_data