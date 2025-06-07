from sentence_transformers import SentenceTransformer

# Modell inicializálása
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def embed_text(text):
    """
    Szöveg beágyazása vektorba.
    
    Paraméterek:
        text (str): A beágyazandó szöveg
        
    Visszatérési érték:
        numpy.ndarray: A beágyazott vektor
    """
    return model.encode(text)