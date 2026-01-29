"""Qdrant vector store client."""

import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from embed_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    UPSERT_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector store wrapper for academic paper embeddings."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Connected to Qdrant at {host}:{port}")

    def init_collection(
        self,
        vector_size: int = EMBEDDING_DIMENSION,
        recreate: bool = False,
    ) -> bool:
        """
        Initialize the collection for storing vectors.

        Args:
            vector_size: Dimension of the embedding vectors
            recreate: If True, delete and recreate the collection

        Returns:
            True if collection was created/exists, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists and recreate:
                logger.warning(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                exists = False

            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Collection created with vector size {vector_size}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            return False

    def upsert_batch(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> int:
        """
        Upsert vectors with payloads in batches.

        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            ids: Optional list of IDs (generated if not provided)
            batch_size: Number of points to upsert in each batch

        Returns:
            Number of points upserted
        """
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError("ids must have the same length as vectors")

        total_upserted = 0

        # Process in batches
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
                for point_id, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                total_upserted += len(points)
                logger.debug(f"Upserted batch {i // batch_size + 1}, total: {total_upserted}")
            except Exception as e:
                logger.error(f"Failed to upsert batch starting at index {i}: {e}")
                raise

        return total_upserted

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            filter: Optional filter conditions (e.g., {"section_type": "methodology"})
            score_threshold: Optional minimum similarity score

        Returns:
            List of search results with payload and score
        """
        # Build filter if provided
        query_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            query_filter = models.Filter(must=conditions)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in results.points
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except UnexpectedResponse:
            return {"name": self.collection_name, "exists": False}
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def delete_by_filter(self, filter: Dict[str, Any]) -> bool:
        """
        Delete points matching a filter.

        Args:
            filter: Filter conditions (e.g., {"paper_name": "some_paper"})

        Returns:
            True if successful
        """
        conditions = []
        for key, value in filter.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=conditions)
                ),
            )
            logger.info(f"Deleted points matching filter: {filter}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return False

    def count_points(self) -> int:
        """Get the total number of points in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
