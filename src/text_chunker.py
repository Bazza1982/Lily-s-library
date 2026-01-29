"""Text chunking utilities using tiktoken and RecursiveCharacterTextSplitter."""

import logging
from typing import List, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embed_config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Initialize tiktoken encoder for token counting
# cl100k_base is used by text-embedding-004
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(_encoder.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into chunks based on token count.

    Args:
        text: The text to split
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Create a splitter that uses tiktoken for length calculation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)
    logger.debug(f"Split text into {len(chunks)} chunks")

    return chunks


def chunk_text_with_metadata(
    text: str,
    paper_name: str,
    section_type: str,
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[str, dict]]:
    """
    Split text into chunks and attach metadata to each chunk.

    Args:
        text: The text to split
        paper_name: Name of the paper
        section_type: Type of section (e.g., 'methodology')
        file_path: Original file path
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks

    Returns:
        List of tuples (chunk_text, metadata_dict)
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    result = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "paper_name": paper_name,
            "section_type": section_type,
            "file_path": file_path,
            "chunk_id": i,
            "total_chunks": len(chunks),
        }
        result.append((chunk, metadata))

    return result


def estimate_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> int:
    """
    Estimate the number of chunks a text will produce.

    Args:
        text: The text to estimate
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Estimated number of chunks
    """
    if not text:
        return 0

    total_tokens = count_tokens(text)

    if total_tokens <= chunk_size:
        return 1

    effective_chunk_size = chunk_size - chunk_overlap
    return max(1, (total_tokens - chunk_overlap) // effective_chunk_size + 1)
