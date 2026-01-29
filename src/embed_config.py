"""Configuration for the embedding pipeline."""

import os
from pathlib import Path

# Embedding Model Configuration
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 768

# Text Chunking Configuration
CHUNK_SIZE = 768  # tokens
CHUNK_OVERLAP = 128  # tokens

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "academic_papers")

# Data Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"

# Section Types
SECTION_TYPES = [
    "abstract",
    "introduction",
    "literature_review",
    "methodology",
    "empirical_analysis",
    "conclusion",
]

# Batch Processing
EMBEDDING_BATCH_SIZE = 100  # Number of texts to embed at once
UPSERT_BATCH_SIZE = 100  # Number of vectors to upsert at once

# API Configuration
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
