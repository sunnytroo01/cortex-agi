#!/bin/bash
# =============================================================================
#  CORTEX AGI — One Command Wikipedia Training
#
#  This script handles everything:
#    1. Installs dependencies
#    2. Downloads all of Wikipedia (~20GB, cached after first run)
#    3. Trains Cortex AGI using Hebbian learning on all GPUs
#    4. Checkpoints every 500 steps so no progress is ever lost
#
#  Usage:
#    bash run.sh                                    # auto-detect everything
#    CONFIG=xl bash run.sh                          # override model size
#    CHECKPOINT_DIR=/mnt/storage/ckpts bash run.sh  # custom checkpoint path
#    PASSES=10 bash run.sh                          # more training passes
# =============================================================================
set -e

echo ""
echo "============================================================"
echo "  CORTEX AGI — Wikipedia Training Pipeline"
echo "  Not a transformer. Not backpropagation. Not gradient descent."
echo "  Pure Hebbian learning on cortical columns."
echo "============================================================"
echo ""

# --- Configuration (override with environment variables) ---
CONFIG="${CONFIG:-large}"
PASSES="${PASSES:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"

# --- Step 1: Install dependencies ---
echo "[1/3] Installing dependencies..."
pip install -q torch>=2.0.0 datasets>=2.14.0
echo "  Done"

# --- Step 2: Pre-download Wikipedia ---
echo ""
echo "[2/3] Downloading Wikipedia (6.7M articles, ~20GB)..."
echo "  Cached after first download — instant on subsequent runs"
python3 download_wiki.py
echo "  Done"

# --- Step 3: Detect GPUs and start training ---
echo ""
echo "[3/3] Starting training..."

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "  GPUs detected: $NUM_GPUS"
echo "  Config: $CONFIG"
echo "  Passes: $PASSES"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Checkpoint every: $CHECKPOINT_EVERY steps"
echo ""

TRAIN_ARGS="--config $CONFIG --passes $PASSES --batch-size $BATCH_SIZE --checkpoint-dir $CHECKPOINT_DIR --checkpoint-every $CHECKPOINT_EVERY"

# Resume from latest checkpoint if it exists
if [ -f "$CHECKPOINT_DIR/cortex_latest.pt" ]; then
    echo "  Found existing checkpoint — resuming training"
    TRAIN_ARGS="$TRAIN_ARGS --resume $CHECKPOINT_DIR/cortex_latest.pt"
fi

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "  Multi-GPU mode: $NUM_GPUS GPUs with weight sync"
    echo ""
    torchrun --nproc_per_node=$NUM_GPUS train.py $TRAIN_ARGS
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "  Single GPU mode"
    echo ""
    python3 train.py $TRAIN_ARGS
else
    echo "  WARNING: No GPU detected — training on CPU (very slow)"
    echo "  Using 'small' config for CPU"
    echo ""
    python3 train.py --config small --passes $PASSES --batch-size 32 \
        --checkpoint-dir $CHECKPOINT_DIR --checkpoint-every $CHECKPOINT_EVERY
fi

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Checkpoint: $CHECKPOINT_DIR/cortex_latest.pt"
echo "  Start chat: python3 server.py --checkpoint $CHECKPOINT_DIR/cortex_latest.pt --config $CONFIG"
echo "============================================================"
