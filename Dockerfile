FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (opcional) baixar o modelo de embeddings na build para evitar cold start
# RUN python - << 'PY'
# from sentence_transformers import SentenceTransformer
# SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# PY

COPY . .

EXPOSE 8080
ENV DATA_DIR=/app/data

CMD ["python", "app.py"]
