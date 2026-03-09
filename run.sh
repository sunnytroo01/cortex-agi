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

# --- Auto-detect python command ---
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")

# --- Configuration (override with environment variables) ---
CONFIG="${CONFIG:-large}"
PASSES="${PASSES:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"

# --- Step 1: Install dependencies ---
echo "[1/3] Installing dependencies..."
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt
else
    echo "  WARNING: requirements.txt not found, installing manually"
    pip install -q "torch>=2.0.0" "datasets>=2.14.0"
fi
echo "  Done"

# --- Step 2: Pre-download Wikipedia ---
echo ""
echo "[2/3] Downloading Wikipedia (6.7M articles, ~20GB)..."
echo "  Cached after first download — instant on subsequent runs"
"$PYTHON" download_wiki.py
echo "  Done"

# --- Step 3: Detect GPUs and start training ---
echo ""
echo "[3/3] Starting training..."

NUM_GPUS=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1 || echo "0")
NUM_GPUS="${NUM_GPUS//[^0-9]/}"
NUM_GPUS="${NUM_GPUS:-0}"

echo "  GPUs detected: $NUM_GPUS"
echo "  Config: $CONFIG"
echo "  Passes: $PASSES"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Checkpoint every: $CHECKPOINT_EVERY steps"
echo ""

TRAIN_ARGS=(--config "$CONFIG" --passes "$PASSES" --batch-size "$BATCH_SIZE" --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-every "$CHECKPOINT_EVERY")

# Resume from latest checkpoint if it exists
if [ -f "$CHECKPOINT_DIR/cortex_latest.pt" ]; then
    echo "  Found existing checkpoint — resuming training"
    TRAIN_ARGS+=(--resume "$CHECKPOINT_DIR/cortex_latest.pt")
fi

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "  Multi-GPU mode: $NUM_GPUS GPUs with weight sync"
    echo ""
    torchrun --nproc_per_node="$NUM_GPUS" train.py "${TRAIN_ARGS[@]}"
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "  Single GPU mode"
    echo ""
    "$PYTHON" train.py "${TRAIN_ARGS[@]}"
else
    echo "  WARNING: No GPU detected — training on CPU (very slow)"
    echo "  Overriding to 'small' config and batch-size 32 for CPU"
    echo ""
    "$PYTHON" train.py "${TRAIN_ARGS[@]}" --config small --batch-size 32
fi

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Checkpoint: $CHECKPOINT_DIR/cortex_latest.pt"
echo "  Start chat: $PYTHON server.py --checkpoint $CHECKPOINT_DIR/cortex_latest.pt --config $CONFIG"
echo "============================================================"
