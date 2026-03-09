"""
Cortex AGI — Distributed Training Script

Supports:
- Single GPU training (default)
- Multi-GPU DDP: torchrun --nproc_per_node=N train.py
- Frequent checkpointing (every pass)
- Hybrid training: batched columns + sequential decoder

Usage:
  python train.py                              # small config, local
  python train.py --config medium              # medium config
  python train.py --config large               # B200 single GPU
  torchrun --nproc_per_node=4 train.py --config large  # 4x GPU DDP
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import argparse

from cortex import Cortex, CortexConfig

CHECKPOINT_DIR = "checkpoints"


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
        print(msg)


def get_config(name: str) -> CortexConfig:
    configs = {
        "small": CortexConfig.small,
        "medium": CortexConfig.medium,
        "large": CortexConfig.large,
        "xl": CortexConfig.xl,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from {list(configs.keys())}")
    return configs[name]()


def main():
    parser = argparse.ArgumentParser(description="Train Cortex AGI")
    parser.add_argument("--config", default="small", choices=["small", "medium", "large", "xl"])
    parser.add_argument("--data", default="data/training_corpus.txt")
    parser.add_argument("--batch-passes", type=int, default=30, help="Batched column training passes")
    parser.add_argument("--seq-passes", type=int, default=5, help="Sequential decoder passes")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for batched passes")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save checkpoint every N steps")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    # Load data
    log(rank, f"Loading {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    num_bytes = len(text.encode("utf-8"))
    log(rank, f"  {len(text):,} chars | {num_bytes:,} bytes")

    # For DDP: split text across GPUs
    if world_size > 1:
        chunk_size = len(text) // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank < world_size - 1 else len(text)
        local_text = text[start:end]
        log(rank, f"  GPU {rank}: chars [{start:,}..{end:,}] ({len(local_text):,} chars)")
    else:
        local_text = text

    # Create model
    config = get_config(args.config)
    config.device = device

    if args.resume and os.path.exists(args.resume):
        log(rank, f"  Resuming from {args.resume}...")
        cortex = Cortex.load_checkpoint(args.resume, device=device)
        start_pass = torch.load(args.resume, weights_only=False).get("pass", 0)
    else:
        cortex = Cortex(config).to(device)
        start_pass = 0

    log(rank, f"  Config: {args.config}")
    log(rank, f"  Columns: {config.n_columns} | Regions: {config.n_regions}")
    log(rank, f"  Neurons/col: {config.n_neurons} | Total neurons: {config.n_columns * config.n_neurons:,}")
    log(rank, f"  Parameters: {cortex.num_parameters():,}")
    log(rank, f"  Device: {device} | GPUs: {world_size}")
    log(rank, f"  Sparsity: {config.n_active}/{config.n_neurons} = {config.n_active/config.n_neurons*100:.1f}%")

    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set up step-level checkpointing (every N steps)
    if rank == 0 and args.checkpoint_every > 0:
        def step_checkpoint(c):
            ckpt_path = os.path.join(args.checkpoint_dir, f"cortex_step_{c.step_count:09d}.pt")
            c.save_checkpoint(ckpt_path, extra={"phase": "step"})
            latest_path = os.path.join(args.checkpoint_dir, "cortex_latest.pt")
            c.save_checkpoint(latest_path, extra={"phase": "step"})
            log(rank, f"    [checkpoint] step {c.step_count:,} saved")

        cortex.set_step_callback(step_checkpoint, every=args.checkpoint_every)
        log(rank, f"  Checkpointing every {args.checkpoint_every} steps")

    # Phase 1: Batched column training
    total_passes = args.batch_passes + args.seq_passes
    log(rank, f"\n{'='*60}")
    log(rank, f"  PHASE 1: Batched Training ({args.batch_passes} passes, batch_size={args.batch_size})")
    log(rank, f"{'='*60}")

    for p in range(start_pass + 1, args.batch_passes + 1):
        t0 = time.time()
        acc = cortex.feed_text(local_text, batch_size=args.batch_size)
        elapsed = time.time() - t0
        local_bytes = len(local_text.encode('utf-8'))
        bps = local_bytes / elapsed

        # Sync accuracy across GPUs
        if world_size > 1:
            acc_tensor = torch.tensor([acc], device=device)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
            acc = acc_tensor.item()

        log(rank, f"  Pass {p:3d}/{total_passes} | acc: {acc:.1f}% | "
                   f"{elapsed:.1f}s | {bps:,.0f} B/s | steps: {cortex.step_count:,}")

        # Save checkpoint (rank 0 only)
        if rank == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"cortex_pass_{p:04d}.pt")
            cortex.save_checkpoint(ckpt_path, extra={"pass": p, "phase": "batched"})
            # Also save as "latest"
            latest_path = os.path.join(args.checkpoint_dir, "cortex_latest.pt")
            cortex.save_checkpoint(latest_path, extra={"pass": p, "phase": "batched"})

    # Phase 2: Sequential decoder refinement
    log(rank, f"\n{'='*60}")
    log(rank, f"  PHASE 2: Sequential Decoder Refinement ({args.seq_passes} passes)")
    log(rank, f"{'='*60}")

    for p in range(1, args.seq_passes + 1):
        pass_num = args.batch_passes + p
        t0 = time.time()
        acc = cortex.feed_text(local_text)  # sequential
        elapsed = time.time() - t0
        local_bytes = len(local_text.encode('utf-8'))
        bps = local_bytes / elapsed

        if world_size > 1:
            acc_tensor = torch.tensor([acc], device=device)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
            acc = acc_tensor.item()

        log(rank, f"  Pass {pass_num:3d}/{total_passes} | acc: {acc:.1f}% | "
                   f"{elapsed:.1f}s | {bps:,.0f} B/s | steps: {cortex.step_count:,}")

        if rank == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"cortex_pass_{pass_num:04d}.pt")
            cortex.save_checkpoint(ckpt_path, extra={"pass": pass_num, "phase": "sequential"})
            latest_path = os.path.join(args.checkpoint_dir, "cortex_latest.pt")
            cortex.save_checkpoint(latest_path, extra={"pass": pass_num, "phase": "sequential"})

    # Generation test
    if rank == 0:
        log(rank, f"\n{'='*60}")
        log(rank, f"  GENERATION TEST")
        log(rank, f"{'='*60}")
        prompts = [
            "Hello", "Who are you", "The sky is", "I think that",
            "Cortex is", "The brain", "1 + 1 =", "What is"
        ]
        for prompt in prompts:
            out = cortex.generate(prompt, max_bytes=100, temperature=0.7)
            safe = out.encode('ascii', errors='replace').decode('ascii')
            log(rank, f'  "{prompt}" -> {safe}')

        log(rank, f"\n  Training complete. Total steps: {cortex.step_count:,}")
        log(rank, f"  Final checkpoint: {args.checkpoint_dir}/cortex_latest.pt")

    cleanup_distributed()


if __name__ == "__main__":
    main()
