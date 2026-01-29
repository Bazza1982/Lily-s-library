#!/usr/bin/env python3
"""CLI for local BGE-M3 embedding system.

使用本地部署的 BAAI/bge-m3 模型对论文 chunks 进行向量化。
向量保存到 data/vectors/ 目录，每个论文一个 JSON 文件。

Usage:
    python main_bge_embed.py init     # 扫描待处理文件
    python main_bge_embed.py run      # 执行向量化
    python main_bge_embed.py status   # 显示进度
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import click
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, load_state, save_state


# Default paths
DEFAULT_CHUNKS_DIR = "data/chunks"
DEFAULT_VECTORS_DIR = "data/vectors"
DEFAULT_STATE_FILE = "state/embed_progress.json"
DEFAULT_BATCH_SIZE = 32


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_paper_dirs(chunks_dir: str) -> list[Path]:
    """Get all paper directories from chunks directory."""
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        return []
    return sorted([d for d in chunks_path.iterdir() if d.is_dir()])


def get_paper_name(paper_dir: Path) -> str:
    """Extract paper name from directory path."""
    return paper_dir.name


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--log-level", "-l", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, config, log_level):
    """BGE-M3 Local Embedding System.

    使用本地 BAAI/bge-m3 模型对论文进行向量化，
    支持 GPU (CUDA) 加速和批量处理。
    """
    ctx.ensure_object(dict)
    setup_logging(log_level)

    # Load config if exists
    try:
        ctx.obj["config"] = load_config(config)
    except FileNotFoundError:
        ctx.obj["config"] = {}

    # Set paths with defaults
    paths = ctx.obj["config"].get("paths", {})
    ctx.obj["chunks_dir"] = paths.get("output_dir", DEFAULT_CHUNKS_DIR)
    ctx.obj["vectors_dir"] = paths.get("vectors_dir", DEFAULT_VECTORS_DIR)
    ctx.obj["state_file"] = DEFAULT_STATE_FILE


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize: scan chunks directory and build pending list."""
    chunks_dir = ctx.obj["chunks_dir"]
    state_file = ctx.obj["state_file"]

    click.echo("Initializing BGE-M3 embedding system...")
    click.echo(f"Chunks directory: {chunks_dir}")

    # Get all paper directories
    paper_dirs = get_paper_dirs(chunks_dir)
    click.echo(f"Found {len(paper_dirs)} papers in {chunks_dir}")

    if not paper_dirs:
        click.echo("No papers found. Run chunking first.", err=True)
        return

    # Load existing state
    state = load_state(state_file)

    # Get already processed papers
    completed_set = set(state.get("completed", []))
    failed_set = set(state.get("failed", []))

    # Build pending list (papers not yet processed)
    pending = []
    for paper_dir in paper_dirs:
        paper_name = get_paper_name(paper_dir)
        if paper_name not in completed_set and paper_name not in failed_set:
            pending.append(paper_name)

    # Update state
    state["pending"] = pending
    save_state(state, state_file)

    click.echo(f"\nState initialized:")
    click.echo(f"  - Pending:   {len(pending)}")
    click.echo(f"  - Completed: {len(completed_set)}")
    click.echo(f"  - Failed:    {len(failed_set)}")
    click.echo(f"State saved to {state_file}")


@cli.command()
@click.option("--batch-size", "-b", default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
@click.option("--limit", "-n", default=0, help="Limit number of papers to process (0 = all)")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without doing it")
@click.option("--include-text/--no-text", default=True, help="Include text in vector files")
@click.pass_context
def run(ctx, batch_size, limit, dry_run, include_text):
    """Run embedding on pending papers."""
    chunks_dir = ctx.obj["chunks_dir"]
    vectors_dir = ctx.obj["vectors_dir"]
    state_file = ctx.obj["state_file"]

    logger = logging.getLogger(__name__)

    # Load state
    state = load_state(state_file)
    pending = state.get("pending", [])

    if not pending:
        click.echo("No pending papers to process. Run 'init' first or check status.")
        return

    # Apply limit
    to_process = pending[:limit] if limit > 0 else pending
    click.echo(f"Processing {len(to_process)} papers...")

    if dry_run:
        click.echo("\nDry run - would process:")
        for name in to_process[:10]:
            click.echo(f"  - {name[:70]}...")
        if len(to_process) > 10:
            click.echo(f"  ... and {len(to_process) - 10} more")
        return

    # Import embedder module (lazy load to avoid model loading on --help)
    from bge_embedder import (
        load_embedder,
        process_paper_chunks,
        save_vectors,
        get_device,
        BGEEmbedderError,
    )

    # Load model
    click.echo(f"\nLoading BGE-M3 model (device: {get_device()})...")
    try:
        load_embedder()
    except BGEEmbedderError as e:
        click.echo(f"Failed to load model: {e}", err=True)
        return

    # Create output directory
    Path(vectors_dir).mkdir(parents=True, exist_ok=True)

    # Process papers
    processed_count = 0
    failed_count = 0
    chunks_path = Path(chunks_dir)

    with tqdm(to_process, desc="Embedding papers", unit="paper") as pbar:
        for paper_name in pbar:
            paper_dir = chunks_path / paper_name
            pbar.set_postfix_str(paper_name[:30])

            if not paper_dir.exists():
                logger.warning(f"Paper directory not found: {paper_dir}")
                state["pending"].remove(paper_name)
                state["failed"].append(paper_name)
                failed_count += 1
                continue

            try:
                # Process paper chunks
                result = process_paper_chunks(
                    paper_dir,
                    batch_size=batch_size
                )

                if not result['chunks']:
                    logger.warning(f"No valid chunks in {paper_name}")
                    state["pending"].remove(paper_name)
                    state["failed"].append(paper_name)
                    failed_count += 1
                    continue

                # Save vectors
                save_vectors(
                    result,
                    vectors_dir,
                    include_text=include_text
                )

                # Update state
                state["pending"].remove(paper_name)
                state["completed"].append(paper_name)
                processed_count += 1

            except BGEEmbedderError as e:
                logger.error(f"Embedding error for {paper_name}: {e}")
                state["pending"].remove(paper_name)
                state["failed"].append(paper_name)
                failed_count += 1

            except Exception as e:
                logger.exception(f"Unexpected error for {paper_name}")
                state["pending"].remove(paper_name)
                state["failed"].append(paper_name)
                failed_count += 1

            # Save state periodically (every 10 papers)
            if (processed_count + failed_count) % 10 == 0:
                save_state(state, state_file)

    # Final state save
    save_state(state, state_file)

    click.echo(f"\nEmbedding complete!")
    click.echo(f"  - Processed: {processed_count}")
    click.echo(f"  - Failed:    {failed_count}")
    click.echo(f"  - Remaining: {len(state['pending'])}")
    click.echo(f"Vectors saved to: {vectors_dir}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show current processing status."""
    state_file = ctx.obj["state_file"]
    vectors_dir = ctx.obj["vectors_dir"]

    state = load_state(state_file)

    pending = state.get("pending", [])
    completed = state.get("completed", [])
    failed = state.get("failed", [])
    total = len(pending) + len(completed) + len(failed)

    click.echo("BGE-M3 Embedding Status")
    click.echo("=" * 40)
    click.echo(f"Pending:   {len(pending)}")
    click.echo(f"Completed: {len(completed)}")
    click.echo(f"Failed:    {len(failed)}")

    if total > 0:
        progress = len(completed) / total * 100
        click.echo(f"\nProgress:  {progress:.1f}%")

    if state.get("last_updated"):
        click.echo(f"Updated:   {state['last_updated']}")

    # Check vectors directory
    vectors_path = Path(vectors_dir)
    if vectors_path.exists():
        vector_files = list(vectors_path.glob("*.json"))
        click.echo(f"\nVector files: {len(vector_files)}")

    # Show device info
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Device:    {device}")
    if device == "cuda":
        click.echo(f"GPU:       {torch.cuda.get_device_name(0)}")

    # Show failed papers if any
    if failed:
        click.echo(f"\nFailed papers ({len(failed)}):")
        for name in failed[:5]:
            click.echo(f"  - {name[:60]}...")
        if len(failed) > 5:
            click.echo(f"  ... and {len(failed) - 5} more")


@cli.command("retry-failed")
@click.option("--limit", "-n", default=0, help="Limit number to retry (0 = all)")
@click.pass_context
def retry_failed(ctx, limit):
    """Move failed papers back to pending for retry."""
    state_file = ctx.obj["state_file"]

    state = load_state(state_file)
    failed = state.get("failed", [])

    if not failed:
        click.echo("No failed papers to retry.")
        return

    # Move failed to pending
    to_retry = failed[:limit] if limit > 0 else failed.copy()

    for paper_name in to_retry:
        state["failed"].remove(paper_name)
        state["pending"].append(paper_name)

    save_state(state, state_file)
    click.echo(f"Moved {len(to_retry)} papers from failed to pending.")
    click.echo("Run 'run' to process them.")


@cli.command("list-failed")
@click.pass_context
def list_failed(ctx):
    """List all failed papers."""
    state_file = ctx.obj["state_file"]

    state = load_state(state_file)
    failed = state.get("failed", [])

    if not failed:
        click.echo("No failed papers.")
        return

    click.echo(f"Failed papers ({len(failed)}):")
    for name in failed:
        click.echo(f"  {name}")


@cli.command("process-one")
@click.argument("paper_name")
@click.option("--batch-size", "-b", default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
@click.pass_context
def process_one(ctx, paper_name, batch_size):
    """Process a single paper by name."""
    chunks_dir = ctx.obj["chunks_dir"]
    vectors_dir = ctx.obj["vectors_dir"]
    state_file = ctx.obj["state_file"]

    chunks_path = Path(chunks_dir)
    paper_dir = chunks_path / paper_name

    if not paper_dir.exists():
        click.echo(f"Paper directory not found: {paper_dir}", err=True)
        return

    # Import embedder module
    from bge_embedder import (
        load_embedder,
        process_paper_chunks,
        save_vectors,
        get_device,
        BGEEmbedderError,
    )

    click.echo(f"Processing: {paper_name}")
    click.echo(f"Device: {get_device()}")

    try:
        # Load model
        load_embedder()

        # Process paper
        result = process_paper_chunks(paper_dir, batch_size=batch_size)

        if not result['chunks']:
            click.echo("No valid chunks found.", err=True)
            return

        # Save vectors
        output_file = save_vectors(result, vectors_dir)

        click.echo(f"\nResults:")
        click.echo(f"  - Sections: {len(result['chunks'])}")
        for chunk in result['chunks']:
            click.echo(f"    - {chunk['section']}: {len(chunk['text'])} chars")
        click.echo(f"  - Vector dimension: {len(result['chunks'][0]['vector'])}")
        click.echo(f"  - Saved to: {output_file}")

        # Update state
        state = load_state(state_file)
        if paper_name in state.get("pending", []):
            state["pending"].remove(paper_name)
        if paper_name in state.get("failed", []):
            state["failed"].remove(paper_name)
        if paper_name not in state.get("completed", []):
            state["completed"].append(paper_name)
        save_state(state, state_file)

    except BGEEmbedderError as e:
        click.echo(f"Embedding error: {e}", err=True)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)


@cli.command()
@click.pass_context
def info(ctx):
    """Show model and system information."""
    import torch

    click.echo("BGE-M3 Model Information")
    click.echo("=" * 40)
    click.echo(f"Model:           BAAI/bge-m3")
    click.echo(f"Vector dimension: 1024")
    click.echo(f"Max sequence:    8192 tokens")
    click.echo()
    click.echo("System Information")
    click.echo("-" * 40)
    click.echo(f"PyTorch version: {torch.__version__}")
    click.echo(f"CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        click.echo(f"CUDA version:    {torch.version.cuda}")
        click.echo(f"GPU:             {torch.cuda.get_device_name(0)}")
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        click.echo(f"GPU Memory:      {memory:.1f} GB")


@cli.command()
@click.pass_context
def reset(ctx):
    """Reset state: clear all progress tracking."""
    state_file = ctx.obj["state_file"]

    if click.confirm("This will clear all progress. Continue?"):
        state = {
            "pending": [],
            "completed": [],
            "failed": [],
            "last_updated": datetime.now().isoformat()
        }
        save_state(state, state_file)
        click.echo("State reset. Run 'init' to rescan papers.")


if __name__ == "__main__":
    cli()
