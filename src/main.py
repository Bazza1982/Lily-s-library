#!/usr/bin/env python3
"""CLI for academic paper smart chunking system."""

import sys
import logging
from pathlib import Path
from datetime import datetime

import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config,
    load_state,
    save_state,
    get_paper_files,
    get_paper_name,
)
from chunker import (
    configure_gemini,
    process_paper,
    ChunkerError,
    APIError,
    ParseError,
)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """Academic Paper Smart Chunking System.

    Use Gemini 2.5 Pro to identify section boundaries in academic papers
    and extract them as separate chunks.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["config"] = load_config(config)
    setup_logging(ctx.obj["config"].get("processing", {}).get("log_level", "INFO"))


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the chunking system by scanning input files."""
    config = ctx.obj["config"]
    paths = config["paths"]

    click.echo("Initializing chunking system...")

    # Get all paper files
    paper_files = get_paper_files(paths["input_dir"])
    click.echo(f"Found {len(paper_files)} papers in {paths['input_dir']}")

    # Load existing state
    state = load_state(paths["state_file"])

    # Get already processed papers
    completed_set = set(state.get("completed", []))
    failed_set = set(state.get("failed", []))

    # Build pending list (papers not yet processed)
    pending = []
    for paper_path in paper_files:
        paper_name = get_paper_name(paper_path)
        if paper_name not in completed_set and paper_name not in failed_set:
            pending.append(paper_name)

    # Update state
    state["pending"] = pending
    save_state(state, paths["state_file"])

    click.echo(f"State initialized:")
    click.echo(f"  - Pending: {len(pending)}")
    click.echo(f"  - Completed: {len(completed_set)}")
    click.echo(f"  - Failed: {len(failed_set)}")
    click.echo(f"State saved to {paths['state_file']}")


@cli.command()
@click.option("--limit", "-l", default=0, help="Limit number of papers to process (0 = all)")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without doing it")
@click.pass_context
def run(ctx, limit, dry_run):
    """Run the chunking process on pending papers."""
    config = ctx.obj["config"]
    paths = config["paths"]
    batch_size = config.get("processing", {}).get("batch_size", 10)

    # Load state
    state = load_state(paths["state_file"])
    pending = state.get("pending", [])

    if not pending:
        click.echo("No pending papers to process. Run 'init' first or check status.")
        return

    # Apply limit
    to_process = pending[:limit] if limit > 0 else pending
    click.echo(f"Processing {len(to_process)} papers...")

    if dry_run:
        click.echo("Dry run - would process:")
        for name in to_process[:10]:
            click.echo(f"  - {name}")
        if len(to_process) > 10:
            click.echo(f"  ... and {len(to_process) - 10} more")
        return

    # Configure Gemini
    try:
        configure_gemini(api_key_env=config["gemini"]["api_key_env"])
    except APIError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Process papers
    input_dir = Path(paths["input_dir"])
    output_dir = paths["output_dir"]
    logger = logging.getLogger(__name__)

    processed_count = 0
    for i, paper_name in enumerate(to_process, 1):
        paper_path = input_dir / f"{paper_name}.md"

        if not paper_path.exists():
            logger.warning(f"Paper not found: {paper_path}")
            state["pending"].remove(paper_name)
            state["failed"].append(paper_name)
            continue

        try:
            click.echo(f"[{i}/{len(to_process)}] Processing: {paper_name[:60]}...")
            result = process_paper(paper_path, config, output_dir)

            # Update state
            state["pending"].remove(paper_name)
            state["completed"].append(paper_name)
            processed_count += 1

            click.echo(f"  ✓ Found {result['sections_found']} sections")

        except (APIError, ParseError) as e:
            logger.error(f"Error processing {paper_name}: {e}")
            state["pending"].remove(paper_name)
            state["failed"].append(paper_name)
            click.echo(f"  ✗ Failed: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error processing {paper_name}")
            state["pending"].remove(paper_name)
            state["failed"].append(paper_name)
            click.echo(f"  ✗ Unexpected error: {e}")

        # Save state periodically
        if i % batch_size == 0:
            save_state(state, paths["state_file"])
            logger.debug(f"State saved after {i} papers")

    # Final state save
    save_state(state, paths["state_file"])

    click.echo(f"\nCompleted: {processed_count} papers processed")
    click.echo(f"Pending: {len(state['pending'])}")
    click.echo(f"Failed: {len(state['failed'])}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show current processing status."""
    config = ctx.obj["config"]
    paths = config["paths"]

    state = load_state(paths["state_file"])

    click.echo("Chunking System Status")
    click.echo("=" * 40)
    click.echo(f"Pending:   {len(state.get('pending', []))}")
    click.echo(f"Completed: {len(state.get('completed', []))}")
    click.echo(f"Failed:    {len(state.get('failed', []))}")

    total = (
        len(state.get("pending", []))
        + len(state.get("completed", []))
        + len(state.get("failed", []))
    )
    if total > 0:
        progress = len(state.get("completed", [])) / total * 100
        click.echo(f"\nProgress:  {progress:.1f}%")

    if state.get("last_updated"):
        click.echo(f"Updated:   {state['last_updated']}")

    # Show failed papers if any
    failed = state.get("failed", [])
    if failed:
        click.echo(f"\nFailed papers ({len(failed)}):")
        for name in failed[:5]:
            click.echo(f"  - {name}")
        if len(failed) > 5:
            click.echo(f"  ... and {len(failed) - 5} more")


@cli.command("retry-failed")
@click.option("--limit", "-l", default=0, help="Limit number of papers to retry (0 = all)")
@click.pass_context
def retry_failed(ctx, limit):
    """Move failed papers back to pending for retry."""
    config = ctx.obj["config"]
    paths = config["paths"]

    state = load_state(paths["state_file"])
    failed = state.get("failed", [])

    if not failed:
        click.echo("No failed papers to retry.")
        return

    # Move failed to pending
    to_retry = failed[:limit] if limit > 0 else failed.copy()

    for paper_name in to_retry:
        state["failed"].remove(paper_name)
        state["pending"].append(paper_name)

    save_state(state, paths["state_file"])
    click.echo(f"Moved {len(to_retry)} papers from failed to pending.")
    click.echo("Run 'run' to process them.")


@cli.command()
@click.argument("paper_name")
@click.pass_context
def process_one(ctx, paper_name):
    """Process a single paper by name (without .md extension)."""
    config = ctx.obj["config"]
    paths = config["paths"]

    input_dir = Path(paths["input_dir"])
    paper_path = input_dir / f"{paper_name}.md"

    if not paper_path.exists():
        click.echo(f"Paper not found: {paper_path}", err=True)
        return

    # Configure Gemini
    try:
        configure_gemini(api_key_env=config["gemini"]["api_key_env"])
    except APIError as e:
        click.echo(f"Error: {e}", err=True)
        return

    click.echo(f"Processing: {paper_name}")

    try:
        result = process_paper(paper_path, config, paths["output_dir"])
        click.echo(f"\nResults:")
        click.echo(f"  Total lines: {result['total_lines']}")
        click.echo(f"  Sections found: {result['sections_found']}")
        click.echo(f"\nSection boundaries:")
        for section, bounds in result["boundaries"].items():
            if bounds:
                click.echo(f"  {section}: lines {bounds['start_line']}-{bounds['end_line']}")
            else:
                click.echo(f"  {section}: not found")

        # Update state
        state = load_state(paths["state_file"])
        if paper_name in state.get("pending", []):
            state["pending"].remove(paper_name)
        if paper_name in state.get("failed", []):
            state["failed"].remove(paper_name)
        if paper_name not in state.get("completed", []):
            state["completed"].append(paper_name)
        save_state(state, paths["state_file"])

    except ChunkerError as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.pass_context
def list_failed(ctx):
    """List all failed papers."""
    config = ctx.obj["config"]
    paths = config["paths"]

    state = load_state(paths["state_file"])
    failed = state.get("failed", [])

    if not failed:
        click.echo("No failed papers.")
        return

    click.echo(f"Failed papers ({len(failed)}):")
    for name in failed:
        click.echo(f"  {name}")


if __name__ == "__main__":
    cli()
