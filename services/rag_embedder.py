import os
from typing import List, Any

import numpy as np

from config import settings
from utils.logger import get_logger

logger = get_logger("rag_embedder")

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _flatten_metadata_values(metadata: Any) -> str:
    parts = []

    def _walk(node):
        if node is None:
            return
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
            return
        if isinstance(node, list):
            for value in node:
                _walk(value)
            return
        parts.append(str(node))

    _walk(metadata)
    return " ".join(parts)


def build_embedding_text(document_type: Any, metadata: Any, full_text: Any, max_chars: int = 8000) -> str:
    doc_type = "" if document_type is None else str(document_type)
    meta_text = _flatten_metadata_values(metadata) if metadata is not None else ""
    body = "" if full_text is None else str(full_text)
    combined = f"{doc_type} {meta_text} {body}".strip()
    if max_chars and len(combined) > max_chars:
        return combined[:max_chars]
    return combined


class RagEmbedder:
    _model = None
    _model_name = settings.get_string("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    @classmethod
    def available(cls) -> bool:
        return SentenceTransformer is not None

    @classmethod
    def _get_model(cls):
        if not cls.available():
            return None
        if cls._model is None:
            logger.info(f"[RAG] Loading embedding model: {cls._model_name}")
            cls._model = SentenceTransformer(cls._model_name)
        return cls._model

    @classmethod
    def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        model = cls._get_model()
        if model is None:
            return []
        cleaned = [t if isinstance(t, str) else str(t or "") for t in texts]
        vectors = model.encode(cleaned, show_progress_bar=False, normalize_embeddings=True)
        if isinstance(vectors, list):
            return vectors
        if isinstance(vectors, np.ndarray):
            return vectors.tolist()
        return []

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        vectors = cls.embed_texts([text])
        if vectors:
            return vectors[0]
        return []

