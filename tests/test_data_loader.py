import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.bhgy_cluster.data_loader import load_data, load_subset

@pytest.fixture
def sample_data():
    """Tesztadatok létrehozása."""
    return pd.DataFrame({
        'text': ['Teszt szöveg 1', 'Teszt szöveg 2', 'Teszt szöveg 3'],
        'JogTerulet': ['Polgári', 'Büntető', 'Polgári']
    })

def test_load_data():
    """Teszteli az adatok betöltését."""
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'text' in data.columns
    assert 'JogTerulet' in data.columns

def test_load_subset():
    """Teszteli az adatok részhalmazának betöltését."""
    n_samples = 100
    data = load_subset(n_samples=n_samples, random_state=42)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) <= n_samples
    assert 'text' in data.columns
    assert 'JogTerulet' in data.columns
    assert 'Embedding' in data.columns
    assert all(isinstance(emb, np.ndarray) for emb in data['Embedding'])

def test_load_subset_embeddings():
    """Teszteli a beágyazások helyességét."""
    data = load_subset(n_samples=10, random_state=42)
    
    # Ellenőrizzük, hogy minden beágyazás megfelelő dimenziójú
    embedding_dim = len(data['Embedding'].iloc[0])
    assert all(len(emb) == embedding_dim for emb in data['Embedding'])
