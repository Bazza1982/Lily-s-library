#!/usr/bin/env python3
"""CLI for academic paper embedding and vector indexing system."""

import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime

import click
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from embed_config import (
    CHUNKS_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    UPSERT_BATCH_SIZE,
)
from data_loader import load_sections, count_sections
from text_chunker import chunk_text
from embedder import configure_embedder, get_embeddings, get_query_embedding
from vector_store import QdrantStore


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("--log-level", "-l", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, log_level):
    """Academic Paper Embedding and Vector Indexing System.
    
    Embed academic paper sections and store in Qdrant for semantic search.
    """
    ctx.ensure_object(dict)
    setup_logging(log_level)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize Qdrant collection."""
    click.echo("Initializing Qdrant collection...")
    
    try:
        store = QdrantStore()
        store.init_collection(EMBEDDING_DIMENSION)
        click.echo(f"✓ Collection '{COLLECTION_NAME}' initialized (dimension: {EMBEDDING_DIMENSION})")
    except Exception as e:
        click.echo(f"✗ Failed to initialize collection: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--batch-size", "-b", default=EMBEDDING_BATCH_SIZE, help="Embedding batch size")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without doing it")
@click.pass_context
def run(ctx, batch_size, dry_run):
    """Run the embedding pipeline on all sections."""
    logger = logging.getLogger(__name__)
    
    # Count total sections
    click.echo("Scanning sections...")
    total_sections = count_sections(CHUNKS_DIR)
    click.echo(f"Found {total_sections} sections to process")
    
    if dry_run:
        click.echo("Dry run - no changes made.")
        return
    
    # Initialize
    try:
        configure_embedder()
        store = QdrantStore()
        store.init_collection(EMBEDDING_DIMENSION)
    except Exception as e:
        click.echo(f"✗ Initialization failed: {e}", err=True)
        sys.exit(1)
    
    # Process sections
    points_batch = []
    processed_chunks = 0
    failed_sections = 0
    
    with tqdm(total=total_sections, desc="Processing sections") as pbar:
        for paper_name, section_type, content, file_path in load_sections(CHUNKS_DIR):
            try:
                # Secondary chunking
                chunks = chunk_text(content)
                
                for chunk_id, chunk_content in enumerate(chunks):
                    payload = {
                        "paper_name": paper_name,
                        "section_type": section_type,
                        "file_path": file_path,
                        "chunk_id": chunk_id,
                        "indexed_at": datetime.now().isoformat(),
                    }
                    points_batch.append({
                        "payload": payload,
                        "text": chunk_content,
                    })
                
                # Process batch when full
                if len(points_batch) >= batch_size:
                    _process_batch(store, points_batch, logger)
                    processed_chunks += len(points_batch)
                    points_batch = []
                    
            except Exception as e:
                logger.error(f"Failed to process {paper_name}/{section_type}: {e}")
                failed_sections += 1
            
            pbar.update(1)
    
    # Process remaining batch
    if points_batch:
        _process_batch(store, points_batch, logger)
        processed_chunks += len(points_batch)
    
    click.echo(f"\n✓ Embedding complete!")
    click.echo(f"  - Processed chunks: {processed_chunks}")
    click.echo(f"  - Failed sections: {failed_sections}")


def _process_batch(store: QdrantStore, points_batch: list, logger) -> None:
    """Process and upload a batch of points."""
    texts = [p["text"] for p in points_batch]
    payloads = [p["payload"] for p in points_batch]
    ids = [str(uuid.uuid4()) for _ in points_batch]
    
    try:
        vectors = get_embeddings(texts)
        store.upsert_batch(vectors=vectors, payloads=payloads, ids=ids)
        logger.debug(f"Upserted batch of {len(points_batch)} points")
    except Exception as e:
        logger.error(f"Failed to process batch: {e}")
        raise


@cli.command()
@click.pass_context
def status(ctx):
    """Show collection status."""
    try:
        store = QdrantStore()
        info = store.get_collection_info()
        
        click.echo(f"Collection: {COLLECTION_NAME}")
        click.echo(f"  - Vectors count: {info.get('vectors_count', 'N/A')}")
        click.echo(f"  - Points count: {info.get('points_count', 'N/A')}")
        click.echo(f"  - Status: {info.get('status', 'N/A')}")
    except Exception as e:
        click.echo(f"✗ Failed to get status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--section", "-s", multiple=True, help="Filter by section type")
@click.pass_context
def search(ctx, query, limit, section):
    """Search for relevant paper sections."""
    try:
        configure_embedder()
        store = QdrantStore()
        
        # Get query embedding
        query_vector = get_query_embedding(query)
        
        # Build filter
        filter_dict = None
        if section:
            filter_dict = {"section_type": list(section)}
        
        # Search
        results = store.search(
            query_vector=query_vector,
            limit=limit,
            filter=filter_dict,
        )
        
        click.echo(f"\nSearch results for: '{query}'")
        click.echo("-" * 50)
        
        for i, result in enumerate(results, 1):
            payload = result['payload']
            score = result['score']
            
            click.echo(f"\n{i}. [{score:.3f}] {payload['paper_name']}")
            click.echo(f"   Section: {payload['section_type']}")
            click.echo(f"   Chunk: {payload['chunk_id']}")
            
    except Exception as e:
        click.echo(f"✗ Search failed: {e}", err=True)
        sys.exit(1)


@cli.command("search-rerank")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of final results")
@click.option("--initial", "-i", default=50, help="Initial vector search results")
@click.option("--section", "-s", multiple=True, help="Filter by section type")
@click.pass_context
def search_rerank(ctx, query, limit, initial, section):
    """Search with reranking (two-stage retrieval)."""
    from search_pipeline import SearchPipeline
    
    try:
        configure_embedder()
        pipeline = SearchPipeline()
        
        section_types = list(section) if section else None
        
        click.echo(f"\nSearching with rerank: '{query}'")
        click.echo(f"Stage 1: Vector search (top {initial})")
        click.echo(f"Stage 2: Rerank to top {limit}")
        click.echo("-" * 50)
        
        results = pipeline.search(
            query=query,
            top_k=limit,
            initial_k=initial,
            section_types=section_types,
            use_rerank=True
        )
        
        for i, result in enumerate(results, 1):
            payload = result['payload']
            vector_score = result.get('vector_score', 0)
            rerank_score = result.get('rerank_score', 0)
            
            click.echo(f"\n{i}. {payload['paper_name']}")
            click.echo(f"   Section: {payload['section_type']}")
            click.echo(f"   Vector: {vector_score:.3f} → Rerank: {rerank_score:.3f}")
            
    except Exception as e:
        click.echo(f"✗ Search failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
