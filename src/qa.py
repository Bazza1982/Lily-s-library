#!/usr/bin/env python3
"""Question-Answering with RAG (Retrieval-Augmented Generation).

Usage:
    python qa.py "What is the role of audit in carbon markets?"
"""

import argparse
import os
import logging
from pathlib import Path

import google.generativeai as genai
from search import AcademicSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class AcademicQA:
    """Question-Answering system for academic papers."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize QA system."""
        self.searcher = AcademicSearch()
        self.model = genai.GenerativeModel(model_name)
        
    def answer(self, question: str, top_k: int = 5, language: str = "en") -> dict:
        """
        Answer a question using retrieved paper chunks.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            language: Response language ("en" or "zh")
            
        Returns:
            Dict with answer and sources
        """
        # Retrieve relevant chunks
        logger.info(f"Retrieving relevant papers for: {question}")
        results = self.searcher.search(question, top_k=20, use_rerank=True, final_k=top_k)
        
        if not results:
            return {
                "answer": "No relevant papers found.",
                "sources": [],
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[Source {i}] {r['paper']} - {r['section']}:\n{r['text']}")
            sources.append({
                "paper": r['paper'],
                "section": r['section'],
                "score": r.get('rerank_score', r['vector_score']),
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        lang_instruction = "回答请使用中文。" if language == "zh" else "Please answer in English."
        
        prompt = f"""You are an expert academic research assistant. Based on the following excerpts from academic papers, answer the user's question.

{lang_instruction}

**Guidelines:**
- Only use information from the provided sources
- Cite sources using [Source N] format
- If the sources don't contain enough information, say so
- Be precise and academic in tone
- Synthesize information across sources when relevant

**Retrieved Sources:**

{context}

**Question:** {question}

**Answer:**"""

        # Generate answer
        logger.info("Generating answer with Gemini...")
        response = self.model.generate_content(prompt)
        
        return {
            "question": question,
            "answer": response.text,
            "sources": sources,
            "model": self.model.model_name,
        }


def main():
    parser = argparse.ArgumentParser(description='Ask questions about academic papers')
    parser.add_argument('question', help='Your question')
    parser.add_argument('--top-k', type=int, default=5, help='Number of sources to use')
    parser.add_argument('--zh', action='store_true', help='Answer in Chinese')
    parser.add_argument('--model', default='gemini-2.5-flash', help='Gemini model to use')
    
    args = parser.parse_args()
    
    qa = AcademicQA(model_name=args.model)
    result = qa.answer(
        args.question,
        top_k=args.top_k,
        language="zh" if args.zh else "en",
    )
    
    print(f"\n{'='*60}")
    print(f"❓ Question: {result['question']}")
    print(f"{'='*60}")
    print(f"\n📝 Answer:\n")
    print(result['answer'])
    print(f"\n{'='*60}")
    print(f"📚 Sources ({len(result['sources'])}):")
    for i, s in enumerate(result['sources'], 1):
        print(f"  [{i}] {s['paper']} ({s['section']}) - score: {s['score']:.4f}")


if __name__ == "__main__":
    main()
