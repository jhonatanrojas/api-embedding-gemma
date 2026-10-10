from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from PIL import Image
import io
import base64
import httpx
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('api-embedding-gemma')

app = FastAPI(
    title='EmbeddingGemma 2 API',
    description='Microservicio de inferencia de embeddings multimodal (texto e imagen) basado en google/embeddinggemma-2 (768 dimensiones).',
    version='2.0.0'
)

MODEL_ID = os.getenv('EMBEDDINGGEMMA_MODEL', 'google/embeddinggemma-2')
CACHE_DIR = os.getenv('HF_CACHE_DIR', '/var/lib/hf-cache/st-cache')
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f'Iniciando carga de modelo: {MODEL_ID} en dispositivo: {DEVICE} (cache: {CACHE_DIR})...')

try:
    # Desactivamos únicamente audio_config para ahorrar ~300M de parámetros mientras mantenemos visión activa
    model = SentenceTransformer(
        MODEL_ID,
        device=DEVICE,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        config_kwargs={'audio_config': None},
        model_kwargs={'torch_dtype': torch.float32},
    )
    DIMENSIONS = model.get_sentence_embedding_dimension() if hasattr(model, 'get_sentence_embedding_dimension') else 768
    logger.info(f'✅ Modelo {MODEL_ID} cargado exitosamente. Dimensión vectorial compartida: {DIMENSIONS}d')
except Exception as e:
    logger.critical(f'❌ Error crítico cargando modelo {MODEL_ID}: {e}', exc_info=True)
    raise e


commercial_anchors = [
    'Classification: foto de producto en venta de tienda e-commerce',
    'Classification: prenda de ropa calzado o accesorio comercial',
    'Classification: articulo o servicio de catalogo comercial'
]
rejection_anchors = [
    'Classification: meme de internet captura de pantalla de chat o texto plano',
    'Classification: persona sin ropa o desnudez explicita inapropiada',
    'Classification: fotografia borrosa irrelevante o basura visual'
]

cached_comm_embs = None
cached_rej_embs = None

def get_moderation_anchors():
    global cached_comm_embs, cached_rej_embs
    if cached_comm_embs is None or cached_rej_embs is None:
        cached_comm_embs = model.encode(commercial_anchors, normalize_embeddings=True, show_progress_bar=False)
        cached_rej_embs = model.encode(rejection_anchors, normalize_embeddings=True, show_progress_bar=False)
    return cached_comm_embs, cached_rej_embs


# Helper para cargar y sanitizar imagen (optimizado para CPU)
def load_image(image_url: Optional[str] = None, image_base64: Optional[str] = None) -> Image.Image:
    if not image_url and not image_base64:
        raise HTTPException(status_code=400, detail='Debe proporcionar image_url o image_base64')

    try:
        if image_url:
            headers = {"User-Agent": "Mozilla/5.0"}
            with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
                res = client.get(image_url)
                res.raise_for_status()
                raw_bytes = res.content
        else:
            b64_str = str(image_base64 or '').strip()
            if ',' in b64_str:
                b64_str = b64_str.split(',', 1)[1]
            raw_bytes = base64.b64decode(b64_str)

        img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        # Limitar resolución a máx 512px para inferencia ágil y eficiente en CPU
        max_dim = 512
        if max(img.width, img.height) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return img
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error procesando imagen: {e}', exc_info=True)
        raise HTTPException(status_code=400, detail=f'No se pudo decodificar o descargar la imagen: {str(e)}')


# Modelos Pydantic para Texto
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


# Modelos Pydantic para Imagen & Visión Multimodal
class EmbedImageRequest(BaseModel):
    image_url: Optional[str] = Field(None, description='URL accesible de la imagen a vectorizar.')
    image_base64: Optional[str] = Field(None, description='Imagen en Base64 (con o sin prefijo data:image/...).')
    normalize: Optional[bool] = Field(True, description='Normalizar a norma L2 para similitud coseno.')


class EmbedImageResponse(BaseModel):
    embedding: List[float] = Field(..., description='Vector de 768 dimensiones para la imagen.')
    dims: int = Field(..., description='Dimensión del vector (768).')
    model: str = Field(..., description='Identificador del modelo.')


class ZeroShotCandidate(BaseModel):
    label: str = Field(..., description='Etiqueta evaluada.')
    score: float = Field(..., description='Probabilidad normalizada (Softmax).')
    similarity: float = Field(..., description='Similitud coseno directa (-1.0 a 1.0).')


class ClassifyZeroShotRequest(BaseModel):
    image_url: Optional[str] = Field(None, description='URL de la imagen.')
    image_base64: Optional[str] = Field(None, description='Imagen en Base64.')
    candidate_labels: List[str] = Field(
        ...,
        description='Categorías candidatas para clasificar la imagen (ej: ["Vestido", "Calzado", "Camisa"]).'
    )
    hypothesis_template: Optional[str] = Field(
        'Una foto de {label}',
        description='Plantilla contextual para las etiquetas textuales.'
    )
    temperature: Optional[float] = Field(0.1, description='Temperatura para el cálculo de Softmax.')


class ClassifyZeroShotResponse(BaseModel):
    labels: List[ZeroShotCandidate] = Field(..., description='Candidatos ordenados por probabilidad descendente.')
    top_label: str = Field(..., description='Etiqueta ganadora.')
    model: str = Field(..., description='Identificador del modelo.')


class ModerateImageRequest(BaseModel):
    image_url: Optional[str] = Field(None, description='URL de la imagen.')
    image_base64: Optional[str] = Field(None, description='Imagen en Base64.')


class ModerateImageResponse(BaseModel):
    is_commercial: bool = Field(..., description='True si la imagen es apta y afín a un catálogo comercial.')
    commercial_score: float = Field(..., description='Puntuación de afinidad comercial.')
    rejection_reason: Optional[str] = Field(None, description='Motivo de rechazo en caso de no ser comercial.')
    model: str = Field(..., description='Identificador del modelo.')


# Endpoints de Diagnóstico y Salud
@app.get('/health', tags=['Health'])
def health():
    return {
        'status': 'ok',
        'model': MODEL_ID,
        'device': DEVICE,
        'dims': DIMENSIONS,
        'multimodal': True
    }


@app.get('/info', tags=['Metadata'])
def info():
    return {
        'model_id': MODEL_ID,
        'dims': DIMENSIONS,
        'device': DEVICE,
        'cache_dir': CACHE_DIR,
        'multimodal_vision': True,
        'prompts': list(model.prompts.keys()) if hasattr(model, 'prompts') else []
    }


# Endpoints de Inferencia
@app.post('/embed', response_model=EmbedResponse, tags=['Embeddings'])
def embed(req: EmbedRequest):
    if not req.inputs:
        return EmbedResponse(embeddings=[], dims=DIMENSIONS, model=MODEL_ID)

    try:
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


@app.post('/embed-image', response_model=EmbedImageResponse, tags=['Multimodal'])
def embed_image(req: EmbedImageRequest):
    img = load_image(image_url=req.image_url, image_base64=req.image_base64)
    try:
        embedding = model.encode(
            {'image': img},
            normalize_embeddings=req.normalize,
            show_progress_bar=False
        )
        return EmbedImageResponse(
            embedding=embedding.tolist(),
            dims=DIMENSIONS,
            model=MODEL_ID
        )
    except Exception as e:
        logger.error(f'Error en inferencia /embed-image: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/classify-zero-shot', response_model=ClassifyZeroShotResponse, tags=['Multimodal'])
def classify_zero_shot(req: ClassifyZeroShotRequest):
    if not req.candidate_labels:
        raise HTTPException(status_code=400, detail='candidate_labels no puede estar vacío.')

    img = load_image(image_url=req.image_url, image_base64=req.image_base64)
    try:
        img_emb = model.encode({'image': img}, normalize_embeddings=True, show_progress_bar=False)

        # Preparar hipótesis textuales con prefijo Classification
        template = req.hypothesis_template or 'Una foto de {label}'
        formatted_labels = [f"Classification: {template.format(label=l)}" for l in req.candidate_labels]
        label_embs = model.encode(formatted_labels, normalize_embeddings=True, show_progress_bar=False)

        # Calcular similitudes coseno
        sims = []
        for i in range(len(req.candidate_labels)):
            val = float(np.dot(img_emb, label_embs[i]))
            sims.append(val)

        sims_arr = np.array(sims)
        temp = max(req.temperature or 0.1, 1e-4)
        exp_sims = np.exp((sims_arr - np.max(sims_arr)) / temp)
        scores = exp_sims / np.sum(exp_sims)

        candidates = []
        for i, label in enumerate(req.candidate_labels):
            candidates.append(ZeroShotCandidate(
                label=label,
                score=round(float(scores[i]), 4),
                similarity=round(float(sims_arr[i]), 4)
            ))

        # Ordenar por score descendente
        candidates.sort(key=lambda c: c.score, reverse=True)

        return ClassifyZeroShotResponse(
            labels=candidates,
            top_label=candidates[0].label if candidates else '',
            model=MODEL_ID
        )
    except Exception as e:
        logger.error(f'Error en clasificación zero-shot: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/moderate-image', response_model=ModerateImageResponse, tags=['Multimodal'])
def moderate_image(req: ModerateImageRequest):
    img = load_image(image_url=req.image_url, image_base64=req.image_base64)
    try:
        img_emb = model.encode({'image': img}, normalize_embeddings=True, show_progress_bar=False)

        commercial_anchors = [
            'Classification: foto de producto en venta de tienda e-commerce',
            'Classification: prenda de ropa calzado o accesorio comercial',
            'Classification: articulo o servicio de catalogo comercial'
        ]
        rejection_anchors = [
            'Classification: meme de internet captura de pantalla de chat o texto plano',
            'Classification: persona sin ropa o desnudez explicita inapropiada',
            'Classification: fotografia borrosa irrelevante o basura visual'
        ]

        comm_embs, rej_embs = get_moderation_anchors()

        max_comm_sim = float(max([np.dot(img_emb, c) for c in comm_embs]))
        max_rej_sim = float(max([np.dot(img_emb, r) for r in rej_embs]))

        # Umbral defensivo para e-commerce
        is_commercial = (max_comm_sim >= 0.30) and (max_comm_sim >= max_rej_sim)
        rejection_reason = None
        if not is_commercial:
            if max_rej_sim > max_comm_sim:
                rejection_reason = 'Contenido no comercial o fuera de catálogo detectado.'
            else:
                rejection_reason = 'La imagen no presenta suficientes rasgos de producto comercial.'

        return ModerateImageResponse(
            is_commercial=is_commercial,
            commercial_score=round(max_comm_sim, 4),
            rejection_reason=rejection_reason,
            model=MODEL_ID
        )
    except Exception as e:
        logger.error(f'Error en moderación de imagen: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
