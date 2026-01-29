"""File I/O utilities for the academic paper chunking system."""

import json
import yaml
import re
from pathlib import Path
from typing import Any
from datetime import datetime


def clean_unicode_ligatures(text: str) -> str:
    """
    Clean PDF parser Unicode placeholders like /uniFB01 (fi ligature).
    These placeholders break text semantics and cause LLM processing failures.
    
    Examples:
        /uniFB01 -> fi (ﬁ)
        /uniFB02 -> fl (ﬂ)
        /uni03B1 -> α
    """
    def replace_match(match):
        hex_code = match.group(1)
        try:
            # Convert hex code to Unicode character
            return chr(int(hex_code, 16))
        except ValueError:
            # If conversion fails, return original
            return match.group(0)
    
    # Match /uni followed by 4 hex characters
    pattern = r'/uni([0-9a-fA-F]{4})'
    return re.sub(pattern, replace_match, text)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(state_path: str) -> dict:
    """Load processing state from JSON file."""
    path = Path(state_path)
    if not path.exists():
        return {
            "pending": [],
            "completed": [],
            "failed": [],
            "last_updated": None
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict, state_path: str) -> None:
    """Save processing state to JSON file."""
    state["last_updated"] = datetime.now().isoformat()
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def read_markdown(file_path: str) -> str:
    """Read markdown file content and clean Unicode ligature placeholders."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Clean Unicode placeholders that break LLM processing
    return clean_unicode_ligatures(content)


def prepend_line_numbers(text: str) -> str:
    """
    Prepend line numbers to each line of text.
    Format: "0001: content"
    """
    lines = text.split("\n")
    numbered_lines = []
    for i, line in enumerate(lines, start=1):
        numbered_lines.append(f"{i:04d}: {line}")
    return "\n".join(numbered_lines)


def extract_section_by_lines(text: str, start_line: int, end_line: int) -> str:
    """
    Extract a section from text based on line numbers.
    Lines are 1-indexed (matching the prepended line numbers).
    end_line is inclusive.
    """
    lines = text.split("\n")
    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line  # end_line is inclusive, so we don't subtract 1

    if start_idx < 0:
        start_idx = 0
    if end_idx > len(lines):
        end_idx = len(lines)

    return "\n".join(lines[start_idx:end_idx])


def get_paper_files(input_dir: str) -> list[Path]:
    """Get all markdown files from input directory."""
    input_path = Path(input_dir)
    return sorted(input_path.glob("*.md"))


def get_paper_name(file_path: Path) -> str:
    """Extract paper name from file path (without extension)."""
    return file_path.stem


def save_chunk(output_dir: str, paper_name: str, section_name: str, content: str) -> Path:
    """
    Save a chunk to the output directory.
    Creates: output_dir/paper_name/section_name.txt
    """
    output_path = Path(output_dir) / paper_name
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_file = output_path / f"{section_name}.txt"
    with open(chunk_file, "w", encoding="utf-8") as f:
        f.write(content)

    return chunk_file


def count_lines(text: str) -> int:
    """Count the number of lines in text."""
    return len(text.split("\n"))


def format_file_size(file_path: Path) -> str:
    """Format file size in human-readable format."""
    size = file_path.stat().st_size
    for unit in ["B", "KB", "MB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
