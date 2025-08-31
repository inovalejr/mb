# Base mais leve
FROM python:3.11-slim

# Setar diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos do projeto
COPY requirements.txt .
COPY . .

# Instalar dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expor porta usada pelo Fly.io
EXPOSE 8080

# Rodar app.py
CMD ["python", "app.py"]
