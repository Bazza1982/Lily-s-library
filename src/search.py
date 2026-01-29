#!/usr/bin/env python3
"""Unified semantic search with reranking.

Usage:
    python search.py "your query" [--top-k 10] [--rerank] [--answer]
"""

import argparse
import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel, FlagReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_PATH = Path(__file__).parent.parent / "qdrant_data"
COLLECTION_NAME = "academic_papers"


class AcademicSearch:
    """Academic paper search with semantic retrieval and reranking."""
    
    def __init__(self):
        """Initialize search components."""
        self.client = None
        self.embed_model = None
        self.reranker = None
        
    def _ensure_client(self):
        """Lazy load Qdrant client."""
        if self.client is None:
            logger.info("Connecting to Qdrant...")
            self.client = QdrantClient(path=str(QDRANT_PATH))
            
    def _ensure_embed_model(self):
        """Lazy load embedding model."""
        if self.embed_model is None:
            logger.info("Loading BGE-M3 embedding model...")
            self.embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
            
    def _ensure_reranker(self):
        """Lazy load reranker model."""
        if self.reranker is None:
            logger.info("Loading BGE Reranker...")
            self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True, device='cuda')
    
    def search(self, query: str, top_k: int = 20, use_rerank: bool = True, final_k: int = 5):
        """
        Search for relevant paper chunks.
        
        Args:
            query: Search query
            top_k: Number of candidates to retrieve
            use_rerank: Whether to rerank results
            final_k: Final number of results to return
            
        Returns:
            List of search results with scores
        """
        self._ensure_client()
        self._ensure_embed_model()
        
        # Embed query
        logger.info(f"Searching: {query}")
        query_embedding = self.embed_model.encode(
            [query], 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        query_vector = query_embedding['dense_vecs'][0].tolist()
        
        # Vector search
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points
        
        # Extract results
        candidates = []
        for hit in results:
            candidates.append({
                'paper': hit.payload.get('paper_name', 'Unknown'),
                'section': hit.payload.get('section', 'Unknown'),
                'text': hit.payload.get('text', ''),
                'vector_score': hit.score,
            })
        
        # Rerank if requested
        if use_rerank and candidates:
            self._ensure_reranker()
            logger.info(f"Reranking {len(candidates)} candidates...")
            
            # Prepare pairs for reranking
            pairs = [[query, c['text']] for c in candidates]
            rerank_scores = self.reranker.compute_score(pairs, normalize=True)
            
            # Handle single result case
            if isinstance(rerank_scores, float):
                rerank_scores = [rerank_scores]
            
            # Add rerank scores
            for i, score in enumerate(rerank_scores):
                candidates[i]['rerank_score'] = score
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return candidates[:final_k]
    
    def format_results(self, results, show_text: bool = True):
        """Format results for display."""
        output = []
        for i, r in enumerate(results, 1):
            score_str = f"rerank={r.get('rerank_score', 0):.4f}" if 'rerank_score' in r else f"vector={r['vector_score']:.4f}"
            output.append(f"\n{i}. [{score_str}] {r['paper']}")
            output.append(f"   Section: {r['section']}")
            if show_text:
                preview = r['text'][:300].replace('\n', ' ')
                output.append(f"   {preview}...")
        return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='Search academic papers')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--top-k', type=int, default=20, help='Candidates to retrieve')
    parser.add_argument('--final-k', type=int, default=5, help='Final results to show')
    parser.add_argument('--no-rerank', action='store_true', help='Disable reranking')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    searcher = AcademicSearch()
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
        final_k=args.final_k,
    )
    
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\n🔍 Query: \"{args.query}\"")
        print(f"   Mode: {'Vector + Rerank' if not args.no_rerank else 'Vector only'}")
        print(searcher.format_results(results))


if __name__ == "__main__":
    main()
