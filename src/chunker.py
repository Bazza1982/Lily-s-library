"""Gemini API integration for academic paper section identification."""

import os
import json
import logging
from typing import Optional
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

from utils import (
    prepend_line_numbers,
    extract_section_by_lines,
    save_chunk,
    count_lines,
)

logger = logging.getLogger(__name__)

# Module-level client
_client: Optional[genai.Client] = None


class ChunkerError(Exception):
    """Base exception for chunker errors."""
    pass


class APIError(ChunkerError):
    """API-related errors."""
    pass


class ParseError(ChunkerError):
    """JSON parsing errors from Gemini response."""
    pass


def configure_gemini(api_key: Optional[str] = None, api_key_env: str = "GEMINI_API_KEY") -> None:
    """Configure Gemini API with the provided key."""
    global _client
    key = api_key or os.environ.get(api_key_env)
    if not key:
        raise APIError(f"No API key provided. Set {api_key_env} environment variable.")
    _client = genai.Client(api_key=key)


def build_prompt(sections: list[dict]) -> str:
    """Build the system prompt for section identification."""
    section_list = "\n".join(
        f"  - {s['name']}: {s['description']}" for s in sections
    )

    return f"""You are an expert at analyzing academic papers. Your task is to identify section boundaries in the provided paper.

The paper has been preprocessed with line numbers in the format "NNNN: content" at the start of each line.

Identify the following sections:
{section_list}

IMPORTANT RULES:
1. Return ONLY a JSON object, no other text
2. For each section found, provide the start_line and end_line (both inclusive, 1-indexed)
3. If a section is not found, set its value to null
4. Sections should not overlap
5. The end_line of one section should be the line before the start_line of the next section
6. Include all content that belongs to each section (headers, paragraphs, tables, figures, etc.)
7. For literature_review: This may be called "Literature Review", "Related Work", "Theoretical Background", "Prior Research", or similar
8. For methodology: This may be called "Method", "Methods", "Methodology", "Research Design", "Data", "Sample", or similar
9. For empirical_analysis: This may be called "Results", "Findings", "Analysis", "Empirical Results", "Discussion", or similar

Return JSON in this exact format:
{{
  "abstract": {{"start_line": N, "end_line": M}} or null,
  "introduction": {{"start_line": N, "end_line": M}} or null,
  "literature_review": {{"start_line": N, "end_line": M}} or null,
  "methodology": {{"start_line": N, "end_line": M}} or null,
  "empirical_analysis": {{"start_line": N, "end_line": M}} or null,
  "conclusion": {{"start_line": N, "end_line": M}} or null
}}"""


def create_retry_decorator(config: dict):
    """Create a tenacity retry decorator based on config."""
    retry_config = config.get("retry", {})

    return retry(
        stop=stop_after_attempt(retry_config.get("max_attempts", 3)),
        wait=wait_exponential(
            min=retry_config.get("wait_min", 2),
            max=retry_config.get("wait_max", 10),
        ),
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, InternalServerError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def call_gemini(
    numbered_text: str,
    config: dict,
) -> dict:
    """
    Call Gemini API to identify section boundaries.

    Args:
        numbered_text: Paper text with prepended line numbers
        config: Configuration dictionary

    Returns:
        Dictionary with section boundaries
    """
    global _client
    if _client is None:
        raise APIError("Gemini client not configured. Call configure_gemini() first.")

    gemini_config = config.get("gemini", {})
    sections = config.get("sections", [])

    system_prompt = build_prompt(sections)
    user_prompt = f"Analyze this academic paper and identify section boundaries:\n\n{numbered_text}"

    # Create retry decorator dynamically
    retry_decorator = create_retry_decorator(config)

    @retry_decorator
    def _make_request():
        response = _client.models.generate_content(
            model=gemini_config.get("model", "gemini-2.5-pro"),
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=gemini_config.get("temperature", 0.1),
                max_output_tokens=gemini_config.get("max_output_tokens", 4096),
                response_mime_type="application/json",
            ),
        )
        return response.text

    try:
        response_text = _make_request()
        return parse_gemini_response(response_text)
    except (ResourceExhausted, ServiceUnavailable, InternalServerError) as e:
        raise APIError(f"API error after retries: {e}")
    except Exception as e:
        raise APIError(f"Unexpected API error: {e}")


def parse_gemini_response(response_text: str) -> dict:
    """Parse and validate Gemini's JSON response."""
    try:
        # Check for empty response
        if response_text is None:
            raise ParseError("Gemini returned empty response (None)")
        
        # Clean up response if needed
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result = json.loads(text)

        # Validate structure
        expected_sections = ["abstract", "introduction", "literature_review",
                            "methodology", "empirical_analysis", "conclusion"]

        for section in expected_sections:
            if section not in result:
                result[section] = None
            elif result[section] is not None:
                if not isinstance(result[section], dict):
                    raise ParseError(f"Section {section} must be dict or null")
                if "start_line" not in result[section] or "end_line" not in result[section]:
                    raise ParseError(f"Section {section} missing start_line or end_line")
                # Ensure integers
                result[section]["start_line"] = int(result[section]["start_line"])
                result[section]["end_line"] = int(result[section]["end_line"])

        return result

    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON response: {e}\nResponse: {response_text[:500]}")


def process_paper(
    paper_path: Path,
    config: dict,
    output_dir: str,
) -> dict:
    """
    Process a single paper: identify sections and save chunks.

    Args:
        paper_path: Path to the markdown paper
        config: Configuration dictionary
        output_dir: Directory to save chunks

    Returns:
        Dictionary with processing results
    """
    paper_name = paper_path.stem
    logger.info(f"Processing: {paper_name}")

    # Read original text
    with open(paper_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    total_lines = count_lines(original_text)
    logger.debug(f"Paper has {total_lines} lines")

    # Prepend line numbers
    numbered_text = prepend_line_numbers(original_text)

    # Call Gemini
    boundaries = call_gemini(numbered_text, config)

    # Extract and save sections
    sections_saved = []
    for section_name, bounds in boundaries.items():
        if bounds is None:
            logger.debug(f"Section '{section_name}' not found")
            continue

        start = bounds["start_line"]
        end = bounds["end_line"]

        # Validate bounds
        if start < 1 or end > total_lines or start > end:
            logger.warning(
                f"Invalid bounds for {section_name}: {start}-{end} (total: {total_lines})"
            )
            continue

        # Extract content
        content = extract_section_by_lines(original_text, start, end)

        # Save chunk
        chunk_path = save_chunk(output_dir, paper_name, section_name, content)
        sections_saved.append({
            "section": section_name,
            "start_line": start,
            "end_line": end,
            "path": str(chunk_path),
        })
        logger.debug(f"Saved {section_name}: lines {start}-{end}")

    return {
        "paper_name": paper_name,
        "total_lines": total_lines,
        "sections_found": len(sections_saved),
        "sections": sections_saved,
        "boundaries": boundaries,
    }
