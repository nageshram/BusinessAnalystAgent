"""
Embedding manager — Singleton for embedding model and vector store.

Uses OpenAI-compatible embeddings via OpenRouter or ChromaDB's default.
Avoids re-creating the embedding model on every call.
"""

import logging
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache()
def get_embeddings():
    """
    Get cached embedding model instance.

    Uses OpenAI embeddings via langchain-openai if API key is available,
    otherwise falls back to ChromaDB's default embedding function.
    """
    try:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
        )
    except Exception as e:
        logger.warning(f"OpenAI embeddings failed, using ChromaDB default: {e}")
        # ChromaDB has a built-in default embedding function
        return None


@lru_cache()
def get_vectorstore() -> Chroma:
    """Get cached vectorstore instance."""
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    kwargs = {
        "collection_name": settings.CHROMA_COLLECTION_NAME,
        "persist_directory": settings.CHROMA_PERSIST_DIR,
    }

    if embeddings:
        kwargs["embedding_function"] = embeddings

    return Chroma(**kwargs)
