from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('api-embedding-gemma')

app = FastAPI(
    title='EmbeddingGemma 2 API',
    description='Microservicio de inferencia de embeddings de alta fidelidad basado en google/embeddinggemma-2 (768 dimensiones).',
    version='1.0.0'
)

MODEL_ID = os.getenv('EMBEDDINGGEMMA_MODEL', 'google/embeddinggemma-2')
CACHE_DIR = os.getenv('HF_CACHE_DIR', '/var/lib/hf-cache/st-cache')
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f'Iniciando carga de modelo: {MODEL_ID} en dispositivo: {DEVICE} (cache: {CACHE_DIR})...')

try:
    model = SentenceTransformer(
        MODEL_ID,
        device=DEVICE,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        config_kwargs={'audio_config': None, 'vision_config': None},
        model_kwargs={'torch_dtype': torch.float32},
    )
    DIMENSIONS = model.get_sentence_embedding_dimension()
    logger.info(f'✅ Modelo {MODEL_ID} cargado exitosamente. Dimensión vectorial: {DIMENSIONS}d')
except Exception as e:
    logger.critical(f'❌ Error crítico cargando modelo {MODEL_ID}: {e}', exc_info=True)
    raise e


class EmbedRequest(BaseModel):
    inputs: List[str] = Field(
        ...,
        description='Lista de textos para generar embeddings.',
        example=['SearchQuery: smartphone samsung galaxy s24', 'Document: title: iPhone 15 | text: 128GB titanio']
    )
    task: Optional[str] = Field(
        'Document',
        description='Tarea semántica sugerida si el texto no incluye prefijo explícito (SearchQuery, Document, STS).'
    )
    normalize: Optional[bool] = Field(
        True,
        description='Normalizar vectores con norma L2 (recomendado para similitud coseno).'
    )


class EmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description='Matriz de vectores flotantes generados.')
    dims: int = Field(..., description='Dimensión de cada vector (768).')
    model: str = Field(..., description='Identificador del modelo.')


@app.get('/health', tags=['Health'])
def health():
    return {
        'status': 'ok',
        'model': MODEL_ID,
        'device': DEVICE,
        'dims': DIMENSIONS
    }


@app.get('/info', tags=['Metadata'])
def info():
    return {
        'model_id': MODEL_ID,
        'dims': DIMENSIONS,
        'device': DEVICE,
        'cache_dir': CACHE_DIR,
        'prompts': list(model.prompts.keys()) if hasattr(model, 'prompts') else []
    }


@app.post('/embed', response_model=EmbedResponse, tags=['Embeddings'])
def embed(req: EmbedRequest):
    if not req.inputs:
        return EmbedResponse(embeddings=[], dims=DIMENSIONS, model=MODEL_ID)

    try:
        # Pre-procesar prefijos si no están explícitos
        processed_inputs = []
        for text in req.inputs:
            clean = str(text or '').strip()
            if not clean:
                continue
            if req.task and not (clean.startswith('SearchQuery:') or clean.startswith('Document:') or clean.startswith('STS:')):
                if req.task == 'SearchQuery':
                    clean = f'SearchQuery: {clean}'
                elif req.task == 'STS':
                    clean = f'STS: {clean}'
                elif req.task == 'Document':
                    if not clean.startswith('title:'):
                        clean = f'Document: title: none | text: {clean}'
                    else:
                        clean = f'Document: {clean}'
            processed_inputs.append(clean)

        if not processed_inputs:
            return EmbedResponse(embeddings=[], dims=DIMENSIONS, model=MODEL_ID)

        embeddings = model.encode(
            processed_inputs,
            normalize_embeddings=req.normalize,
            show_progress_bar=False
        )

        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dims=DIMENSIONS,
            model=MODEL_ID
        )
    except Exception as e:
        logger.error(f'Error en inferencia /embed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
