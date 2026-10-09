# EmbeddingGemma 2 API 🚀

Microservicio ligero y de alto rendimiento en **FastAPI** para la generación de embeddings de texto utilizando el modelo oficial [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) de Google DeepMind.

Genera representaciones vectoriales densas de **768 dimensiones** con normalización L2, optimizado tanto para entornos CPU con SIMD (AVX2/AVX-512) como aceleración GPU CUDA.

---

## 📌 Características

- **Modelo:** `google/embeddinggemma-2` (270M parámetros de texto, arquitectura de alta fidelidad).
- **Dimensiones:** `768d` (compatible nativo con PostgreSQL `pgvector`, Qdrant, Milvus, Pinecone).
- **Normalización L2:** Salidas pre-normalizadas listas para búsqueda por similitud de cosenos (`<=>`).
- **Soporte de Prefijos Oficiales:**
  - `SearchQuery: <query>` para consultas de búsqueda en tiempo real.
  - `Document: title: <title> | text: <content>` para indexación de catálogo o base de conocimiento.
  - `STS: <text>` para tareas de similitud semántica general.
- **Doble Modo de Despliegue:** Docker Container o Bare-Metal nativo con `systemd`.

---

## 🛠️ Requisitos

- Python 3.10+
- PyTorch 2.1+
- Al menos 2 GB de memoria RAM libre.

---

## 🚀 Instalación y Despliegue

### Opción A: Despliegue con Docker (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/jhonatanrojas/api-embedding-gemma.git
cd api-embedding-gemma

# Levantar con docker-compose
docker compose up -d --build
```

El servicio estará disponible en `http://localhost:8080`.

---

### Opción B: Despliegue Bare-Metal (Systemd en VPS Linux)

```bash
# 1. Crear entorno virtual
sudo mkdir -p /opt/embeddinggemma
sudo chown -R $USER:$USER /opt/embeddinggemma
cd /opt/embeddinggemma
git clone https://github.com/jhonatanrojas/api-embedding-gemma.git .

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt

# 2. Configurar servicio systemd
sudo cp embeddinggemma.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now embeddinggemma

# 3. Verificar estado
sudo systemctl status embeddinggemma
```

---

## 📡 Endpoints de la API

### 1. Health Check
```http
GET /health
```
**Respuesta:**
```json
{
  "status": "ok",
  "model": "google/embeddinggemma-2",
  "device": "cpu",
  "dims": 768
}
```

---

### 2. Metadatos del Modelo
```http
GET /info
```
**Respuesta:**
```json
{
  "model_id": "google/embeddinggemma-2",
  "dims": 768,
  "device": "cpu",
  "prompts": ["SearchQuery", "Document", "STS", ...]
}
```

---

### 3. Generar Embeddings
```http
POST /embed
Content-Type: application/json
```

**Payload:**
```json
{
  "inputs": [
    "SearchQuery: celulares de alta gama",
    "Document: title: iPhone 15 Pro | text: Chip A17 Pro y acabado en titanio"
  ],
  "task": "Document",
  "normalize": true
}
```

**Respuesta:**
```json
{
  "embeddings": [
    [-0.0535, -0.0205, -0.0383, ...],
    [0.0124, 0.0451, -0.0092, ...]
  ],
  "dims": 768,
  "model": "google/embeddinggemma-2"
}
```

---

## 🧪 Ejemplo con cURL

```bash
curl -X POST http://127.0.0.1:8080/embed \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["SearchQuery: zapatillas deportivas running"],
    "task": "SearchQuery"
  }'
```

---

## 📄 Licencia

Este proyecto está bajo la licencia [Apache 2.0](LICENSE), en concordancia con los términos del modelo base de Google Gemma.
