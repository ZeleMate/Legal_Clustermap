# 1. Hivatalos Python báziskép
FROM python:3.12-slim

# 2. Munkakönyvtár létrehozása a konténeren belül
WORKDIR /app

# 3. requirements.txt bemásolása és függőségek telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Az egész projekt bemásolása a konténerbe
COPY . .

# 5. Projekt telepítése fejlesztői módban
RUN pip install -e .

# 6. Tesztadatok létrehozása
RUN mkdir -p /app/data
RUN echo "text,JogTerulet\nTeszt szöveg 1,Polgári\nTeszt szöveg 2,Büntető\nTeszt szöveg 3,Polgári" > /app/data/test_data.csv

# 7. Környezeti változó beállítása
ENV DATA_PATH=/app/data/test_data.csv
ENV PYTHONPATH=/app

# 8. Alapértelmezett parancs – tesztek futtatása
ENTRYPOINT ["pytest"]
CMD ["tests/"]