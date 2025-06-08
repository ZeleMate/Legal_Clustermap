# 1. Hivatalos Python báziskép
FROM python:3.12-slim

# 2. Munkakönyvtár létrehozása a konténeren belül
WORKDIR /app

# 3. requirements.txt bemásolása és függőségek telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Az egész projekt bemásolása a konténerbe
COPY . .

# 5. Alapértelmezett parancs – main.py futtatása
CMD ["python", "main.py"]