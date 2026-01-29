#!/usr/bin/env python3
"""Build Qdrant index from vector files (local mode).

使用本地文件模式，不需要启动 Qdrant 服务。
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
VECTORS_DIR = Path(__file__).parent.parent / "data" / "vectors"
QDRANT_PATH = Path(__file__).parent.parent / "qdrant_data"
COLLECTION_NAME = "academic_papers"
VECTOR_DIMENSION = 1024  # BGE-M3 dimension
BATCH_SIZE = 100


def load_vector_file(file_path: Path) -> Dict[str, Any]:
    """Load a single vector file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_index():
    """Build the Qdrant index from vector files."""
    
    # Initialize local Qdrant client
    logger.info(f"Initializing Qdrant (local mode) at {QDRANT_PATH}")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    
    # Check if collection exists, recreate if needed
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if exists:
        logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
    
    # Create collection
    logger.info(f"Creating collection: {COLLECTION_NAME} (dim={VECTOR_DIMENSION})")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )
    
    # Get all vector files
    vector_files = list(VECTORS_DIR.glob("*.json"))
    logger.info(f"Found {len(vector_files)} vector files")
    
    # Process in batches
    points = []
    point_id = 0
    
    for file_path in tqdm(vector_files, desc="Loading vectors"):
        try:
            data = load_vector_file(file_path)
            paper_name = data.get('paper_name', file_path.stem)
            
            for chunk in data.get('chunks', []):
                section = chunk.get('section', 'unknown')
                text = chunk.get('text', '')
                vector = chunk.get('vector', [])
                
                if not vector:
                    continue
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        'paper_name': paper_name,
                        'section': section,
                        'text': text[:500],  # Truncate for storage
                        'text_length': len(text),
                    }
                ))
                point_id += 1
                
                # Batch upsert
                if len(points) >= BATCH_SIZE:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )
                    points = []
                    
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Final batch
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
    
    # Get collection info
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Index built successfully!")
    logger.info(f"  - Points: {info.points_count}")
    logger.info(f"  - Vectors: {info.vectors_count}")
    
    return info.points_count


if __name__ == "__main__":
    count = build_index()
    print(f"\n✅ Index complete! {count} vectors indexed.")
