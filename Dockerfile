# Usa uma imagem base leve com Python 3.12
FROM python:3.12-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as dependências sem cache
RUN pip install --no-cache-dir -r requirements.txt

# Pré-carrega o modelo da biblioteca sentence-transformers
RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")'

# Define variável de ambiente para o diretório de dados
ENV DATA_DIR=/app/data

# Copia todos os arquivos do projeto para o container
COPY . .

# Expõe a porta 8080 para acesso externo
EXPOSE 8080

# Comando padrão para iniciar a aplicação
CMD ["python", "app.py"]