# Cortex AGI

A brain-inspired artificial general intelligence trained on all of Wikipedia using cortical column architecture and Hebbian learning.

**Not a transformer. Not backpropagation. Not gradient descent.**

Cortex AGI uses the same principles as the biological neocortex:
- **Cortical columns** — the universal computation unit of the brain
- **Hebbian learning** — "neurons that fire together wire together"
- **Sparse distributed representations** — only ~5% of neurons active at any time
- **Predictive coding** — columns predict their input and learn from errors
- **Online learning** — learns from every input, no training/inference split

## One Command Training

Clone the repo on your GPU machine and run:

```bash
git clone https://github.com/sunnytroo01/cortex-agi.git
cd cortex-agi
bash run.sh
```

That's it. The script:
1. Installs dependencies
2. Downloads all of English Wikipedia (6.7M articles, ~20GB, cached after first run)
3. Trains on every GPU it finds using distributed Hebbian learning
4. Checkpoints every 500 steps — no progress is ever lost

## Architecture

```
Input (bytes) → Embedding → Region 1 → Region 2 → ... → Region N → Decoder → Output
                               ↑↓           ↑↓              ↑↓
                           Feedback     Feedback         Feedback
```

Each region contains hundreds of cortical columns processed in parallel. Each column independently learns features through local Hebbian rules — no global loss function needed.

## Training

### Default: Wikipedia via HuggingFace (recommended)

```bash
bash run.sh                                    # auto-detect GPUs, train on Wikipedia
CONFIG=xl PASSES=10 bash run.sh                # override settings
CHECKPOINT_DIR=/mnt/storage/ckpts bash run.sh  # custom checkpoint path
```

### Manual commands

```bash
# Single GPU
python train.py --config large

# Multi-GPU (e.g., 2x B200)
torchrun --nproc_per_node=2 train.py --config large

# Resume from checkpoint
python train.py --resume checkpoints/cortex_latest.pt

# Custom text file
python train.py --data myfile.txt --config small

# Wiki files from wikiextractor
python download_wiki.py --method dump
python train.py --data-dir data/wiki
```

### Multi-GPU Hebbian Learning

Unlike transformer DDP which syncs gradients, Cortex AGI uses **periodic model averaging** for distributed Hebbian learning:
- Each GPU processes a different shard of Wikipedia
- At the end of each pass, weights are averaged across all GPUs
- This gives N-GPU throughput while maintaining a single coherent model

## Model Sizes

| Config | Columns | Neurons | Regions | Chunk Size | Params | GPU Memory |
|--------|---------|---------|---------|------------|--------|------------|
| small  | 32      | 8K      | 4       | 1 (byte)   | ~3M    | < 1 GB     |
| medium | 128     | 49K     | 4       | 16 bytes   | ~30M   | ~1 GB      |
| large  | 256     | 128K    | 4       | 32 bytes   | ~100M  | ~3 GB      |
| xl     | 512     | 512K    | 4       | 64 bytes   | ~800M  | ~20 GB     |

Chunk encoding groups N bytes into one "percept" (like sensory receptive fields in the brain). This makes Wikipedia-scale training 30-60x faster while preserving the same learning principles.

## Checkpointing

Every 500 training steps, a checkpoint is saved automatically. Checkpoints are also saved:
- After every training pass over the full dataset
- On Ctrl+C (graceful interrupt)
- As both numbered (`cortex_pass_0001.pt`) and latest (`cortex_latest.pt`)

Training auto-resumes from the latest checkpoint if one exists.

## Chat UI

After training, start the web server:

```bash
python server.py --checkpoint checkpoints/cortex_latest.pt --config large
# Open http://localhost:5000
```

## How It Works

Unlike transformers which use attention and backpropagation, Cortex AGI learns through:

1. **Forward pass**: Input bytes are embedded and passed through hierarchical regions of cortical columns
2. **k-Winners-Take-All**: Only the top ~5% of neurons in each column activate (sparse coding)
3. **Prediction**: Each column predicts its input based on its activation pattern
4. **Hebbian update**: Connections between co-active neurons are strengthened (local learning rule)
5. **Predictive error**: The difference between prediction and actual input drives learning

This is how the biological brain learns — no teacher signal, no global optimizer, just local synaptic plasticity.

## License

MIT
