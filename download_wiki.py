#!/usr/bin/env python3
"""
Cortex AGI — Wikipedia Data Pipeline

Downloads English Wikipedia for training. Two methods:
  1. HuggingFace datasets (default) — fast, pre-extracted, ~20GB cached
  2. Direct dump + wikiextractor — manual, ~22GB download + hours extraction

Usage:
  python download_wiki.py                    # HuggingFace (recommended)
  python download_wiki.py --method dump      # Wikimedia dump + wikiextractor
"""

import argparse
import os
import sys
import time


def download_huggingface():
    """Download Wikipedia via HuggingFace datasets (recommended)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print("Downloading Wikipedia from HuggingFace Hub...")
    print("  Dataset: wikimedia/wikipedia (20231101.en)")
    print("  Size: ~20GB (cached at ~/.cache/huggingface/datasets/)")
    print("  Articles: ~6.7 million\n")

    t0 = time.time()
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    elapsed = time.time() - t0

    print(f"\n  Done! {len(ds):,} articles downloaded in {elapsed/60:.0f}m")
    print(f"  Cached at: ~/.cache/huggingface/datasets/")
    print(f"  Ready for training: python train.py")


def download_dump():
    """Download Wikipedia dump from Wikimedia and extract with wikiextractor."""
    import subprocess
    import shutil
    import glob

    DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    DUMP_FILE = "data/enwiki-pages-articles.xml.bz2"
    WIKI_DIR = "data/wiki"

    os.makedirs("data", exist_ok=True)

    # Download dump
    if os.path.exists(DUMP_FILE):
        size_gb = os.path.getsize(DUMP_FILE) / (1024 ** 3)
        if size_gb >= 20:
            print(f"  Dump exists: {DUMP_FILE} ({size_gb:.1f} GB)")
        else:
            print(f"  Incomplete dump ({size_gb:.1f} GB), resuming download...")
            _download_file(DUMP_URL, DUMP_FILE)
    else:
        print(f"Downloading Wikipedia dump (~22 GB)...")
        print(f"  URL: {DUMP_URL}")
        _download_file(DUMP_URL, DUMP_FILE)

    # Extract with wikiextractor
    existing = glob.glob(os.path.join(WIKI_DIR, "**", "wiki_*"), recursive=True)
    if existing:
        print(f"  Already extracted: {len(existing)} files in {WIKI_DIR}/")
    else:
        if not shutil.which("wikiextractor"):
            print("ERROR: wikiextractor not found. Run: pip install wikiextractor")
            sys.exit(1)

        print(f"Extracting with wikiextractor (this takes 2-4 hours)...")
        n_cpus = os.cpu_count() or 4
        subprocess.run([
            "wikiextractor", DUMP_FILE,
            "-o", WIKI_DIR,
            "--json",
            "--processes", str(n_cpus),
            "--no-templates",
            "-b", "5M"
        ], check=True)

    n_files = len(glob.glob(os.path.join(WIKI_DIR, "**", "wiki_*"), recursive=True))
    print(f"\n  Done! {n_files} files in {WIKI_DIR}/")
    print(f"  Ready for training: python train.py --data-dir {WIKI_DIR}")


def _download_file(url, dest):
    """Download a file using wget, curl, or Python urllib."""
    import shutil
    import subprocess

    if shutil.which("wget"):
        subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", dest, url], check=True)
    elif shutil.which("curl"):
        subprocess.run(["curl", "-L", "-C", "-", "-o", dest, url], check=True)
    else:
        print("  Using Python urllib (no resume support)...")
        import urllib.request

        def _progress(count, block_size, total_size):
            downloaded = count * block_size
            gb = downloaded / (1024 ** 3)
            if total_size > 0:
                pct = downloaded / total_size * 100
                total_gb = total_size / (1024 ** 3)
                sys.stdout.write(f"\r  {gb:.1f}/{total_gb:.1f} GB ({pct:.1f}%)")
            else:
                sys.stdout.write(f"\r  {gb:.1f} GB downloaded")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()


def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia for Cortex AGI")
    parser.add_argument("--method", default="huggingface",
                        choices=["huggingface", "dump"],
                        help="Download method (default: huggingface)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Cortex AGI — Wikipedia Download")
    print("=" * 60)

    t0 = time.time()
    if args.method == "huggingface":
        download_huggingface()
    else:
        download_dump()

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.0f} minutes")


if __name__ == "__main__":
    main()
