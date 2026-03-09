#!/usr/bin/env python3
"""
Cortex AGI — Wikipedia Training Script

Trains on the full English Wikipedia (6.7M articles, ~20GB) using Hebbian
learning. No gradient descent. No backpropagation. Pure cortical column
learning with sparse distributed representations.

Supports:
- Single GPU: python train.py
- Multi-GPU DDP: torchrun --nproc_per_node=N train.py
- Resume from checkpoint: python train.py --resume checkpoints/cortex_latest.pt
- Custom data: python train.py --data file.txt
- Wiki files: python train.py --data-dir data/wiki

One command to train on Wikipedia:
  ./run.sh
"""

import torch
import torch.distributed as dist
import time
import os
import sys
import argparse
import random
import json
import glob
import signal

from cortex import Cortex, CortexConfig

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, f"cuda:{local_rank}"
    return 0, 1, "cuda" if torch.cuda.is_available() else "cpu"


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def fmt_bytes(n):
    if n >= 1e9: return f"{n/1e9:.1f} GB"
    if n >= 1e6: return f"{n/1e6:.1f} MB"
    if n >= 1e3: return f"{n/1e3:.0f} KB"
    return f"{n} B"


def fmt_time(s):
    if s >= 3600: return f"{s/3600:.1f}h"
    if s >= 60: return f"{s/60:.0f}m"
    return f"{s:.0f}s"


def get_config(name):
    configs = {
        "small": CortexConfig.small,
        "medium": CortexConfig.medium,
        "large": CortexConfig.large,
        "xl": CortexConfig.xl,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}")
    return configs[name]()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikipedia_hf(rank):
    """Load Wikipedia from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        log(rank, "ERROR: 'datasets' package not installed.")
        log(rank, "  Install: pip install datasets")
        log(rank, "  Or use:  python download_wiki.py --method dump")
        log(rank, "           python train.py --data-dir data/wiki")
        sys.exit(1)

    log(rank, "Loading Wikipedia from HuggingFace...")
    log(rank, "  Dataset: wikimedia/wikipedia (20231101.en)")
    log(rank, "  First run downloads ~20GB to cache")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    log(rank, f"  Loaded {len(ds):,} articles")
    return ds


def load_wiki_files(data_dir):
    """Get wikiextractor JSON files from a directory."""
    return sorted(glob.glob(os.path.join(data_dir, "**", "wiki_*"), recursive=True))


def iter_wiki_file(filepath):
    """Yield article text from a wikiextractor JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
                text = article.get("text", "")
                if len(text) > 100:
                    yield text
            except (json.JSONDecodeError, KeyError):
                continue


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------

def train_wikipedia_hf(cortex, args, rank, world_size, device, start_pass):
    """Train on Wikipedia from HuggingFace datasets (default mode)."""
    ds = load_wikipedia_hf(rank)

    # Shard across GPUs
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)
        log(rank, f"  GPU {rank}: {len(ds):,} articles (shard {rank+1}/{world_size})")

    total_articles = len(ds)
    training_start = time.time()

    for pass_num in range(start_pass + 1, args.passes + 1):
        log(rank, f"\n{'='*60}")
        log(rank, f"  PASS {pass_num}/{args.passes}")
        log(rank, f"{'='*60}")

        ds_shuffled = ds.shuffle(seed=pass_num)

        pass_bytes = 0
        pass_weighted_acc = 0.0
        pass_start = time.time()
        articles_done = 0

        for article in ds_shuffled:
            text = article.get("text", "")
            if len(text) < 100:
                continue

            acc = cortex.feed_text(text, batch_size=args.batch_size)
            n_bytes = len(text.encode("utf-8"))
            pass_bytes += n_bytes
            pass_weighted_acc += acc * n_bytes / 100.0
            articles_done += 1

            # Progress every 1000 articles
            if articles_done % 1000 == 0:
                elapsed = time.time() - pass_start
                bps = pass_bytes / max(elapsed, 0.001)
                overall_acc = pass_weighted_acc / max(pass_bytes, 1) * 100.0
                rate = articles_done / elapsed
                eta = (total_articles - articles_done) / rate if rate > 0 else 0
                log(rank,
                    f"  {articles_done:>8,}/{total_articles:,} articles | "
                    f"acc: {overall_acc:5.1f}% | "
                    f"{fmt_bytes(bps)}/s | "
                    f"ETA: {fmt_time(eta)} | "
                    f"steps: {cortex.step_count:,}")

        # --- End of pass ---
        pass_elapsed = time.time() - pass_start
        overall_acc = pass_weighted_acc / max(pass_bytes, 1) * 100.0

        # Sync weights across GPUs (average for Hebbian learning)
        if world_size > 1:
            dist.barrier()
            cortex.sync_weights()
            acc_t = torch.tensor([overall_acc], device=device)
            dist.all_reduce(acc_t, op=dist.ReduceOp.AVG)
            overall_acc = acc_t.item()
            step_t = torch.tensor([cortex.step_count], device=device, dtype=torch.long)
            dist.all_reduce(step_t, op=dist.ReduceOp.SUM)
            cortex.step_count = step_t.item()

        log(rank, f"\n  Pass {pass_num} complete")
        log(rank, f"  Accuracy: {overall_acc:.1f}%")
        log(rank, f"  Data: {fmt_bytes(pass_bytes)} in {fmt_time(pass_elapsed)}")
        log(rank, f"  Throughput: {fmt_bytes(pass_bytes / max(pass_elapsed, 1))}/s")
        log(rank, f"  Total steps: {cortex.step_count:,}")

        if rank == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"cortex_pass_{pass_num:04d}.pt")
            cortex.save_checkpoint(ckpt, extra={"pass": pass_num, "acc": overall_acc})
            latest = os.path.join(args.checkpoint_dir, "cortex_latest.pt")
            cortex.save_checkpoint(latest, extra={"pass": pass_num, "acc": overall_acc})
            log(rank, f"  Saved: {ckpt}")

    _run_generation_test(cortex, rank)
    total_time = time.time() - training_start
    log(rank, f"\n  Training complete. {cortex.step_count:,} steps in {fmt_time(total_time)}")


def train_wiki_files(cortex, args, rank, world_size, device, start_pass=0):
    """Train on wikiextractor output directory."""
    all_files = load_wiki_files(args.data_dir)
    if not all_files:
        log(rank, f"ERROR: No wiki files in {args.data_dir}")
        log(rank, "  Run: python download_wiki.py --method dump")
        sys.exit(1)

    log(rank, f"  Found {len(all_files)} wiki files in {args.data_dir}")
    training_start = time.time()

    for pass_num in range(start_pass + 1, args.passes + 1):
        log(rank, f"\n{'='*60}")
        log(rank, f"  PASS {pass_num}/{args.passes}")
        log(rank, f"{'='*60}")

        my_files = all_files[rank::world_size]
        rng = random.Random(pass_num)
        rng.shuffle(my_files)

        pass_bytes = 0
        pass_weighted_acc = 0.0
        pass_start = time.time()

        for file_idx, filepath in enumerate(my_files):
            for text in iter_wiki_file(filepath):
                acc = cortex.feed_text(text, batch_size=args.batch_size)
                n_bytes = len(text.encode("utf-8"))
                pass_bytes += n_bytes
                pass_weighted_acc += acc * n_bytes / 100.0

            if (file_idx + 1) % 50 == 0:
                elapsed = time.time() - pass_start
                bps = pass_bytes / max(elapsed, 0.001)
                overall_acc = pass_weighted_acc / max(pass_bytes, 1) * 100.0
                log(rank,
                    f"  file {file_idx+1}/{len(my_files)} | "
                    f"acc: {overall_acc:.1f}% | "
                    f"{fmt_bytes(bps)}/s | "
                    f"steps: {cortex.step_count:,}")

        if world_size > 1:
            dist.barrier()
            cortex.sync_weights()

        pass_elapsed = time.time() - pass_start
        overall_acc = pass_weighted_acc / max(pass_bytes, 1) * 100.0
        log(rank, f"  Pass {pass_num}: acc {overall_acc:.1f}% | "
                  f"{fmt_bytes(pass_bytes)} in {fmt_time(pass_elapsed)}")

        if rank == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"cortex_pass_{pass_num:04d}.pt")
            cortex.save_checkpoint(ckpt, extra={"pass": pass_num, "acc": overall_acc})
            latest = os.path.join(args.checkpoint_dir, "cortex_latest.pt")
            cortex.save_checkpoint(latest, extra={"pass": pass_num, "acc": overall_acc})
            log(rank, f"  Saved: {ckpt}")

    _run_generation_test(cortex, rank)
    total_time = time.time() - training_start
    log(rank, f"\n  Training complete. {cortex.step_count:,} steps in {fmt_time(total_time)}")


def train_single_file(cortex, args, rank, world_size, device, start_pass=0):
    """Train on a single text file."""
    log(rank, f"  Loading {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    n_bytes = len(text.encode("utf-8"))
    log(rank, f"  {len(text):,} chars | {fmt_bytes(n_bytes)}")

    if world_size > 1:
        chunk = len(text) // world_size
        start = rank * chunk
        end = start + chunk if rank < world_size - 1 else len(text)
        text = text[start:end]
        log(rank, f"  GPU {rank}: chars [{start:,}..{end:,}]")

    for pass_num in range(start_pass + 1, args.passes + 1):
        t0 = time.time()
        acc = cortex.feed_text(text, batch_size=args.batch_size)
        elapsed = time.time() - t0
        local_bytes = len(text.encode("utf-8"))
        bps = local_bytes / max(elapsed, 0.001)

        if world_size > 1:
            acc_t = torch.tensor([acc], device=device)
            dist.all_reduce(acc_t, op=dist.ReduceOp.AVG)
            acc = acc_t.item()

        log(rank, f"  Pass {pass_num}/{args.passes} | acc: {acc:.1f}% | "
                  f"{elapsed:.1f}s | {bps:,.0f} B/s | steps: {cortex.step_count:,}")

        if rank == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"cortex_pass_{pass_num:04d}.pt")
            cortex.save_checkpoint(ckpt, extra={"pass": pass_num, "acc": acc})
            latest = os.path.join(args.checkpoint_dir, "cortex_latest.pt")
            cortex.save_checkpoint(latest, extra={"pass": pass_num, "acc": acc})

    _run_generation_test(cortex, rank)


def _run_generation_test(cortex, rank):
    """Quick generation test after training."""
    if rank != 0:
        return
    log(rank, f"\n{'='*60}")
    log(rank, f"  GENERATION TEST")
    log(rank, f"{'='*60}")
    prompts = [
        "The", "Science is", "History of", "Mathematics",
        "Water is", "The brain", "Earth is", "In the year",
    ]
    for prompt in prompts:
        out = cortex.generate(prompt, max_bytes=150, temperature=0.7)
        safe = out.encode("ascii", errors="replace").decode("ascii")
        log(rank, f'  "{prompt}" -> {safe}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Cortex AGI")
    parser.add_argument("--config", default="large",
                        choices=["small", "medium", "large", "xl"])
    parser.add_argument("--data", default=None,
                        help="Single text file (overrides default Wikipedia mode)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory of wikiextractor JSON files")
    parser.add_argument("--passes", type=int, default=3,
                        help="Training passes over the data (default: 3)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for column training (default: 256)")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for checkpoints (default: checkpoints)")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save checkpoint every N steps (default: 500)")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    # --- Model ---
    config = get_config(args.config)
    config.device = device
    start_pass = 0

    if args.resume and os.path.exists(args.resume):
        log(rank, f"Resuming from {args.resume}")
        cortex = Cortex.load_checkpoint(args.resume, device=device)
        ckpt_data = torch.load(args.resume, weights_only=False, map_location=device)
        start_pass = ckpt_data.get("pass", 0)
        log(rank, f"  Resumed at pass {start_pass}, step {cortex.step_count:,}")
    else:
        cortex = Cortex(config).to(device)

    log(rank, f"\n{'='*60}")
    log(rank, f"  CORTEX AGI — Hebbian Learning on Wikipedia")
    log(rank, f"{'='*60}")
    log(rank, f"  Config: {args.config} | Params: {cortex.num_parameters():,}")
    log(rank, f"  Columns: {config.n_columns} | Regions: {config.n_regions}")
    log(rank, f"  Neurons/col: {config.n_neurons} | "
              f"Total: {config.n_columns * config.n_neurons:,}")
    log(rank, f"  Sparsity: {config.n_active}/{config.n_neurons} = "
              f"{config.n_active / config.n_neurons * 100:.1f}%")
    log(rank, f"  Device: {device} | GPUs: {world_size}")
    log(rank, f"  Batch size: {args.batch_size} | Passes: {args.passes}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Step-level checkpointing
    if rank == 0 and args.checkpoint_every > 0:
        def _on_step(c):
            path = os.path.join(args.checkpoint_dir, "cortex_latest.pt")
            c.save_checkpoint(path, extra={"step": c.step_count})
        cortex.set_step_callback(_on_step, every=args.checkpoint_every)
        log(rank, f"  Checkpointing every {args.checkpoint_every} steps")

    # Graceful interrupt handler
    def _handle_interrupt(sig, frame):
        log(rank, "\n\n  Interrupted! Saving checkpoint...")
        if rank == 0:
            path = os.path.join(args.checkpoint_dir, "cortex_interrupted.pt")
            cortex.save_checkpoint(path, extra={"step": cortex.step_count})
            log(rank, f"  Saved: {path}")
        cleanup_distributed()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_interrupt)

    # --- Train ---
    if args.data:
        log(rank, f"\n  Mode: single file ({args.data})")
        train_single_file(cortex, args, rank, world_size, device, start_pass)
    elif args.data_dir:
        log(rank, f"\n  Mode: wiki files ({args.data_dir})")
        train_wiki_files(cortex, args, rank, world_size, device, start_pass)
    else:
        log(rank, f"\n  Mode: Wikipedia (HuggingFace datasets)")
        train_wikipedia_hf(cortex, args, rank, world_size, device, start_pass)

    log(rank, f"\n  Final checkpoint: {args.checkpoint_dir}/cortex_latest.pt")
    cleanup_distributed()


if __name__ == "__main__":
    main()
