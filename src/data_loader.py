"""Data loader for academic paper chunks."""

import os
import logging
from pathlib import Path
from typing import Iterator, Tuple

from embed_config import CHUNKS_DIR, SECTION_TYPES

logger = logging.getLogger(__name__)


def load_sections(
    data_path: Path = None,
) -> Iterator[Tuple[str, str, str, str]]:
    """
    Load all section files from the chunks directory.

    Args:
        data_path: Path to the chunks directory. Defaults to CHUNKS_DIR from config.

    Yields:
        Tuple of (paper_name, section_type, content, file_path)
    """
    chunks_dir = Path(data_path) if data_path else CHUNKS_DIR

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return

    # Iterate through paper directories
    paper_dirs = sorted([d for d in chunks_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(paper_dirs)} paper directories")

    for paper_dir in paper_dirs:
        paper_name = paper_dir.name

        # Load each section file
        for section_type in SECTION_TYPES:
            section_file = paper_dir / f"{section_type}.txt"

            if not section_file.exists():
                logger.debug(f"Section not found: {paper_name}/{section_type}")
                continue

            try:
                content = section_file.read_text(encoding="utf-8")

                if not content.strip():
                    logger.debug(f"Empty section: {paper_name}/{section_type}")
                    continue

                file_path = str(section_file.relative_to(CHUNKS_DIR.parent.parent))
                yield paper_name, section_type, content, file_path

            except Exception as e:
                logger.warning(f"Error reading {section_file}: {e}")
                continue


def count_papers(data_path: Path = None) -> int:
    """Count the number of papers in the chunks directory."""
    chunks_dir = Path(data_path) if data_path else CHUNKS_DIR

    if not chunks_dir.exists():
        return 0

    return len([d for d in chunks_dir.iterdir() if d.is_dir()])


def count_sections(data_path: Path = None) -> int:
    """Count the total number of section files."""
    chunks_dir = Path(data_path) if data_path else CHUNKS_DIR

    if not chunks_dir.exists():
        return 0

    count = 0
    for paper_dir in chunks_dir.iterdir():
        if paper_dir.is_dir():
            for section_type in SECTION_TYPES:
                if (paper_dir / f"{section_type}.txt").exists():
                    count += 1

    return count


def get_paper_sections(paper_name: str, data_path: Path = None) -> dict:
    """
    Get all sections for a specific paper.

    Args:
        paper_name: Name of the paper directory
        data_path: Path to the chunks directory

    Returns:
        Dictionary mapping section_type to content
    """
    chunks_dir = Path(data_path) if data_path else CHUNKS_DIR
    paper_dir = chunks_dir / paper_name

    if not paper_dir.exists():
        logger.error(f"Paper directory not found: {paper_name}")
        return {}

    sections = {}
    for section_type in SECTION_TYPES:
        section_file = paper_dir / f"{section_type}.txt"
        if section_file.exists():
            try:
                content = section_file.read_text(encoding="utf-8")
                if content.strip():
                    sections[section_type] = content
            except Exception as e:
                logger.warning(f"Error reading {section_file}: {e}")

    return sections
