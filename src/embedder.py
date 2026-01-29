"""Gemini Embedding module using google-generativeai."""

import os
import logging
from typing import List, Optional

import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from embed_config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    GEMINI_API_KEY_ENV,
)

logger = logging.getLogger(__name__)

# Module-level state
_configured = False


class EmbeddingError(Exception):
    """Embedding-related errors."""
    pass


def configure_embedder(api_key: Optional[str] = None) -> None:
    """
    Configure the Gemini API for embeddings.

    Args:
        api_key: API key. If None, reads from GEMINI_API_KEY environment variable.
    """
    global _configured

    key = api_key or os.environ.get(GEMINI_API_KEY_ENV)
    if not key:
        raise EmbeddingError(
            f"No API key provided. Set {GEMINI_API_KEY_ENV} environment variable."
        )

    genai.configure(api_key=key)
    _configured = True
    logger.info("Gemini embedder configured successfully")


def _ensure_configured() -> None:
    """Ensure the embedder is configured."""
    if not _configured:
        configure_embedder()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _embed_batch(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Internal function to embed a batch of texts with retry logic.

    Args:
        texts: List of texts to embed
        model_name: Name of the embedding model

    Returns:
        List of embedding vectors
    """
    result = genai.embed_content(
        model=f"models/{model_name}",
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def get_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        model_name: Name of the embedding model (default: text-embedding-004)
        batch_size: Number of texts to embed in each API call

    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    _ensure_configured()

    if not texts:
        return []

    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug(f"Embedding batch {i // batch_size + 1}, size: {len(batch)}")

        try:
            embeddings = _embed_batch(batch, model_name)
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(f"Failed to embed batch starting at index {i}: {e}")
            raise EmbeddingError(f"Embedding failed: {e}")

    return all_embeddings


def get_query_embedding(
    query: str,
    model_name: str = EMBEDDING_MODEL,
) -> List[float]:
    """
    Get embedding for a single query text.

    Uses task_type="retrieval_query" for better search performance.

    Args:
        query: Query text to embed
        model_name: Name of the embedding model

    Returns:
        Embedding vector
    """
    _ensure_configured()

    try:
        result = genai.embed_content(
            model=f"models/{model_name}",
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise EmbeddingError(f"Query embedding failed: {e}")
