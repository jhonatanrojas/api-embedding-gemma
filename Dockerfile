FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar dependencias de Python (utilizando PyTorch CPU para optimizar tamaño de imagen)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8080
ENV HOST=0.0.0.0
ENV EMBEDDINGGEMMA_MODEL=google/embeddinggemma-2
ENV HF_CACHE_DIR=/data/hf-cache

EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["sh", "-c", "uvicorn app:app --host ${HOST} --port ${PORT} --workers 1"]
