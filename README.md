# Legal Clustermap

![CI](https://github.com/ZeleMate/Legal_Clustermap/actions/workflows/ci.yml/badge.svg)

Ez a projekt jogi szövegek automatikus klaszterezését végzi el modern gépi tanulási módszerekkel. A program beágyazza a jogi szövegeket, majd PCA-val csökkenti a dimenziót, automatikusan meghatározza az optimális klaszterszámot, KMeans klaszterezést alkalmaz, végül értékeli és vizualizálja az eredményeket.

## Tartalomjegyzék

- [Funkciók](#funkciók)
- [Telepítés](#telepítés)
- [Használat](#használat)
- [Projekt Struktúra](#projekt-struktúra)
- [Működés](#működés)
- [Kimenetek Értelmezése](#kimenetek-értelmezése)
- [Függőségek](#függőségek)
- [Fejlesztői Információk](#fejlesztői-információk)
- [.env használata](#env-használata)

## Funkciók

- Jogi szövegek automatikus beágyazása (sentence-transformers)
- Dimenziócsökkentés PCA segítségével
- KMeans klaszterezés, az optimális klaszterszám automatikus meghatározásával
- Klaszterezés értékelése több metrikával (homogeneity, completeness, mutual_info, fowlkes_mallows, adjusted_rand, adjusted_mutual_info)
- Eredmények vizualizálása, jogterületek feliratozott legendával
- Részletes statisztikák a klaszterek eloszlásáról

## Telepítés

1. Klónozd le a repository-t:
```bash
git clone https://github.com/yourusername/Legal_Clustermap.git
cd Legal_Clustermap
```

2. Válaszd ki a telepítési módszert:

### Docker használatával (ajánlott)

```bash
# Docker image építése
docker build -t legal-clustermap .

# Docker container futtatása
docker run -it --rm -v $(pwd)/data:/app/data legal-clustermap
```

### Helyi telepítés

1. Hozz létre és aktiválj egy virtuális környezetet:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# vagy
venv\Scripts\activate  # Windows
```

2. Telepítsd a csomagot fejlesztői módban:
```bash
pip install -e .
```

## Használat

1. Készítsd elő az adatfájlt:
   - A bemeneti CSV fájlnak tartalmaznia kell egy "text" oszlopot a szövegekkel
   - Opcionálisan tartalmazhat egy "JogTerulet" oszlopot a valós kategorizációval

2. Futtasd a programot:
```bash
python main.py
```

## Projekt Struktúra

```
.
├── main.py              # Program belépési pont
├── setup.py            # Csomag telepítési konfiguráció
├── src/                # Forráskód könyvtár
│   └── Legal_Clustermap/   # Fő modul
│       ├── data_loader.py    # Adatok betöltése és előfeldolgozása
│       ├── clustering.py     # Klaszterezési logika (PCA, optimális klaszterszám, KMeans)
│       ├── metrics.py        # Klaszterezési metrikák számítása
│       ├── visualization.py  # Eredmények vizualizálása
│       ├── evaluation.py     # Klaszterezés értékelése
│       └── models/          # Modell definíciók
├── tests/              # Tesztek
├── data/              # Adatfájlok
├── utils/             # Segédeszközök
├── Dockerfile         # Docker konfiguráció
├── requirements.txt   # Függőségek
└── README.md          # Dokumentáció
```

## Működés

A program a következő lépéseket végzi:

1. **Adatok betöltése** (`data_loader.py`):
   - CSV fájlból betölti az adatokat
   - Véletlenszerűen kiválaszt 1000 sort (alapértelmezés)
   - Beágyazza a szövegeket (egyszerre, nem batch-ben)

2. **Klaszterezés** (`clustering.py`):
   - PCA-val 2 dimenzióra csökkenti az embeddingeket
   - Automatikusan meghatározza az optimális klaszterszámot több metrika alapján
   - KMeans klaszterezést alkalmaz
   - Debug információkat ír ki (opcionális)

3. **Értékelés és vizualizáció** (`evaluation.py`, `visualization.py`):
   - Kiszámítja a következő metrikákat: homogeneity, completeness, mutual_info, fowlkes_mallows, adjusted_rand, adjusted_mutual_info
   - Klaszterek eloszlását ábrázolja jogterületek szerint, feliratozott legendával
   - t-SNE vizualizáció a klaszterekről és jogterületekről
   - Metrikák ábrázolása
   - Részletes klaszter elemzés és értékelés

## Kimenetek Értelmezése

### Metrikák

- **Homogeneity**: Méri, hogy minden klaszter csak egy osztályból áll-e (0-1)
- **Completeness**: Méri, hogy egy osztály minden eleme ugyanabba a klaszterbe került-e (0-1)
- **Mutual Info**: Kölcsönös információ a klaszterek és osztályok között (0-1)
- **Fowlkes-Mallows**: Klaszterek és osztályok átfedése (0-1)
- **Adjusted Rand**: Klaszterek és osztályok egyezése, véletlenhez viszonyítva (0-1)
- **Adjusted Mutual Info**: Kölcsönös információ normalizált verziója (0-1)

### Vizualizáció

- **Klaszterek eloszlása jogterületek szerint**: Oszlopdiagram, a legendában a jogterületek nevei szerepelnek
- **t-SNE ábrázolás**: Két ábra, az egyik a klasztereket, a másik a jogterületeket mutatja
- **Metrikák ábrázolása**: Oszlopdiagram a főbb metrikákról
- **Klaszter elemzés**: Részletes statisztikák és vizualizációk a klaszterek belső szerkezetéről

## Függőségek

A program a következő főbb függőségeket használja:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- sentence-transformers (a beágyazó modellhez)
- Docker (opcionális, ha Docker-t használsz)

A teljes lista a `requirements.txt` fájlban található.

## Fejlesztői Információk

### Debug Mód

A program debug módban is futtatható, ami részletes információkat ad a feldolgozás minden lépéséről:

```python
kmeans_labels, pca_data, optimal_k = cluster_data(embeddings, debug=True)
```

### Klaszterek Száma

A klaszterek számát a program automatikusan határozza meg a metrikák alapján, de a `clustering.py`-ban a maximum érték paraméterezhető.

### Adatfájl Formátum

A bemeneti CSV fájlnak a következő oszlopokat kell tartalmaznia:
- `text`: A jogi szövegek
- `JogTerulet` (opcionális): A valós kategorizáció

### Fejlesztési Tervek

- [ ] Több klaszterezési algoritmus implementálása
- [ ] Hiperparaméter optimalizálás
- [ ] Webes felület készítése
- [ ] Több értékelési metrika hozzáadása
- [ ] Batch feldolgozás implementálása nagyobb adathalmazokhoz

## .env használata

A program az adatfájl elérési útját a `.env` fájlból olvassa be. Példa `.env` tartalom:

```
DATA_PATH=data/cleaned_data_for_embedding.csv
```

- Másold át a `.env.example` fájlt `.env` néven, és állítsd be a saját adatfájlod elérési útját.
- A `.env` fájlt **ne töltsd fel a GitHub-ra**!
- Ha az adatfájl máshol van, módosítsd a `DATA_PATH` értékét a saját elérési utadra.