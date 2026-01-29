"""
batch_parse.py - Batch PDF parsing with Docling (v2 - with timeout & progress)

Usage:
    python batch_parse.py --start 0 --end 100    # Parse PDFs 0-99
    python batch_parse.py --continue             # Continue from last position
"""

import argparse
import json
import torch
import sys
import signal
from pathlib import Path
from datetime import datetime
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from docling.document_converter import DocumentConverter

# Configuration
LIBRARY_PATH = Path(r"C:\Users\Barry Li (UoN)\OneDrive - The University Of Newcastle\Desktop\AI\Library")
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "parsed"
LOG_PATH = Path(__file__).parent.parent / "data" / "logs"
PROGRESS_FILE = Path(__file__).parent.parent / "data" / "parse_progress.json"

# Timeout per PDF (seconds)
PDF_TIMEOUT = 300  # 5 minutes per PDF


def load_progress():
    """Load parsing progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"parsed": [], "failed": [], "skipped": [], "total_time": 0}


def save_progress(progress):
    """Save parsing progress to file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def convert_with_timeout(converter, pdf_path, timeout_seconds):
    """Convert PDF with timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(converter.convert, pdf_path)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            return None


def parse_batch(start_idx: int, end_idx: int, continue_mode: bool = False):
    """Parse a batch of PDFs."""
    print("=" * 70, flush=True)
    print(f"BATCH PARSING: PDFs {start_idx} to {end_idx - 1}", flush=True)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Timeout per PDF: {PDF_TIMEOUT}s ({PDF_TIMEOUT // 60} min)", flush=True)
    print("=" * 70, flush=True)
    
    # Check GPU
    print(f"\nPyTorch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    
    # Get all PDF files (sorted for consistent ordering)
    all_pdfs = sorted(LIBRARY_PATH.glob("*.pdf"))
    total_pdfs = len(all_pdfs)
    print(f"\nTotal PDFs in library: {total_pdfs}", flush=True)
    
    # Load progress
    progress = load_progress()
    
    # In continue mode, find first unparsed PDF
    if continue_mode:
        parsed_set = set(progress["parsed"]) | set(progress["failed"]) | set(progress.get("skipped", []))
        for i, pdf in enumerate(all_pdfs):
            if pdf.name not in parsed_set:
                start_idx = i
                break
        print(f"Continue mode: Starting from index {start_idx}", flush=True)
    
    # Select batch
    batch_pdfs = all_pdfs[start_idx:end_idx]
    batch_size = len(batch_pdfs)
    print(f"Batch size: {batch_size} PDFs (index {start_idx}-{end_idx - 1})", flush=True)
    
    if not batch_pdfs:
        print("No PDFs to process in this range!", flush=True)
        return {"success": 0, "skipped": 0, "failed": 0, "timeout": 0}
    
    # Create directories
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize converter
    print("\nInitializing Docling DocumentConverter...", flush=True)
    converter = DocumentConverter()
    print("Converter ready!\n", flush=True)
    
    # Ensure skipped list exists
    if "skipped" not in progress:
        progress["skipped"] = []
    
    # Statistics
    stats = {"success": 0, "skipped": 0, "failed": 0, "timeout": 0}
    batch_start_time = time.time()
    
    # Process each PDF
    for i, pdf_path in enumerate(batch_pdfs):
        global_idx = start_idx + i
        short_name = pdf_path.stem[:55] + "..." if len(pdf_path.stem) > 55 else pdf_path.stem
        
        # Real-time progress header
        print(f"\n[{i + 1}/{batch_size}] #{global_idx} | {short_name}", flush=True)
        
        output_file = OUTPUT_PATH / f"{pdf_path.stem}.md"
        
        # Skip if already processed
        if output_file.exists() or pdf_path.name in progress["parsed"]:
            print(f"  → SKIP (already parsed)", flush=True)
            stats["skipped"] += 1
            continue
        
        if pdf_path.name in progress["failed"]:
            print(f"  → SKIP (previously failed)", flush=True)
            stats["skipped"] += 1
            continue
        
        if pdf_path.name in progress.get("skipped", []):
            print(f"  → SKIP (previously timed out)", flush=True)
            stats["skipped"] += 1
            continue
        
        try:
            start_time = time.time()
            print(f"  → Parsing... (timeout: {PDF_TIMEOUT}s)", flush=True)
            
            # Convert with timeout
            result = convert_with_timeout(converter, pdf_path, PDF_TIMEOUT)
            
            if result is None:
                # Timeout occurred
                elapsed = time.time() - start_time
                print(f"  ⏱️ TIMEOUT after {elapsed:.0f}s - skipping", flush=True)
                progress["skipped"].append(pdf_path.name)
                stats["timeout"] += 1
                save_progress(progress)
                continue
            
            if not result.document:
                raise ValueError("Empty result from Docling")
            
            markdown_text = result.document.export_to_markdown()
            elapsed = time.time() - start_time
            
            # Save markdown
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            print(f"  ✅ SUCCESS: {len(markdown_text):,} chars in {elapsed:.1f}s", flush=True)
            
            # Update progress
            progress["parsed"].append(pdf_path.name)
            progress["total_time"] += elapsed
            stats["success"] += 1
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ FAILED: {str(e)[:60]}", flush=True)
            progress["failed"].append(pdf_path.name)
            stats["failed"] += 1
            
            # Log error details
            error_log = LOG_PATH / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pdf_path.stem[:30]}.txt"
            with open(error_log, "w", encoding="utf-8") as f:
                f.write(f"PDF: {pdf_path}\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
        
        # Save progress after each file
        save_progress(progress)
        
        # Print running stats every 10 files
        if (i + 1) % 10 == 0:
            elapsed_min = (time.time() - batch_start_time) / 60
            rate = (stats["success"] + stats["failed"] + stats["timeout"]) / max(elapsed_min, 0.1)
            print(f"\n  📊 Progress: {stats['success']} ok, {stats['failed']} fail, {stats['timeout']} timeout | {rate:.1f}/min\n", flush=True)
    
    # Summary
    batch_elapsed = time.time() - batch_start_time
    print("\n" + "=" * 70, flush=True)
    print(f"BATCH COMPLETE!", flush=True)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Time: {batch_elapsed / 60:.1f} minutes", flush=True)
    print("-" * 70, flush=True)
    print(f"Success:  {stats['success']}", flush=True)
    print(f"Skipped:  {stats['skipped']}", flush=True)
    print(f"Failed:   {stats['failed']}", flush=True)
    print(f"Timeout:  {stats['timeout']}", flush=True)
    print("-" * 70, flush=True)
    print(f"Total parsed:  {len(progress['parsed'])}", flush=True)
    print(f"Total failed:  {len(progress['failed'])}", flush=True)
    print(f"Total skipped: {len(progress.get('skipped', []))}", flush=True)
    print("=" * 70, flush=True)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch PDF parsing with Docling")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--continue", dest="continue_mode", action="store_true", help="Continue from last position")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per PDF in seconds (default: 300)")
    args = parser.parse_args()
    
    global PDF_TIMEOUT
    PDF_TIMEOUT = args.timeout
    
    # Get total PDFs for default end
    all_pdfs = sorted(LIBRARY_PATH.glob("*.pdf"))
    end_idx = args.end if args.end else len(all_pdfs)
    
    stats = parse_batch(args.start, end_idx, args.continue_mode)
    
    # Return non-zero if all failed
    if stats["success"] == 0 and (stats["failed"] > 0 or stats["timeout"] > 0):
        exit(1)


if __name__ == "__main__":
    main()
