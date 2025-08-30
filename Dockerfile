FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# deps nativos básicos (compilar wheels quando necessário)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Pré-baixa o modelo de embeddings p/ evitar cold start
RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")'

ENV DATA_DIR=/app/data \
    OPENAI_MODEL=gpt-4o-mini \
    GEMINI_MODEL=gemini-1.5-flash \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

COPY . .

EXPOSE 8080

# Servir o FastAPI com Gunicorn + UvicornWorker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
