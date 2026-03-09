# Cortex AGI

A brain-inspired artificial general intelligence built on cortical column architecture with Hebbian learning.

**Not a transformer. Not backpropagation. Not gradient descent.**

Cortex AGI uses the same principles as the biological neocortex:
- **Cortical columns** — the universal computation unit of the brain
- **Hebbian learning** — "neurons that fire together wire together"
- **Sparse distributed representations** — only ~5% of neurons active at any time
- **Predictive coding** — columns predict their input and learn from errors
- **Online learning** — learns from every input, no training/inference split

## Architecture

```
Input (bytes) → Embedding → Region 1 → Region 2 → ... → Region N → Decoder → Output
                               ↑↓           ↑↓              ↑↓
                           Feedback     Feedback         Feedback
```

Each region contains hundreds of cortical columns processed in parallel. Each column independently learns features through local Hebbian rules — no global loss function needed.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Build training data (100K English words + grammar rules)
python data/build_corpus.py

# Train (small config for testing)
python train.py --config small

# Train on B200 (single GPU)
python train.py --config large

# Train on multiple B200s
torchrun --nproc_per_node=4 train.py --config large

# Chat UI
python server.py
# Open http://localhost:5000
```

## Model Sizes

| Config | Columns | Neurons | Params | GPU Memory |
|--------|---------|---------|--------|------------|
| small  | 32      | 8K      | ~3M    | < 1 GB     |
| medium | 256     | 128K    | ~50M   | ~2 GB      |
| large  | 1024    | 1M      | ~800M  | ~20 GB     |
| xl     | 2048    | 4M      | ~6B    | ~150 GB    |

## Training Data

The training corpus includes:
- **62,920 unique English words** from Wiktionary frequency lists
- **Comprehensive grammar rules** — parts of speech, tenses, sentence structure, punctuation, word formation, spelling rules
- **Dialogue examples** — conversational patterns
- **High-frequency reinforcement** — common patterns repeated for Hebbian learning

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
