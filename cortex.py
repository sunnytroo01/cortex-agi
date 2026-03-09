"""
Cortex AGI — A Brain-Inspired Cortical Column Network

NOT a transformer. NOT gradient descent. NOT backpropagation.

Architecture inspired by the neocortex:
- Universal 6-layer microcircuit (same algorithm in every column)
- Hebbian learning ("neurons that fire together wire together")
- Sparse distributed representations (~5% activation)
- Predictive coding (columns predict their input, learn from errors)
- Continuous online learning (no training/inference split)
- Hierarchical regions with feedback connections

Designed for NVIDIA B200 GPUs (192GB HBM3e, 2.25 PFLOPS FP16).
Supports single-GPU and multi-GPU (DDP) training.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CortexConfig:
    """Configuration for Cortex AGI.

    Presets:
    - 'small':  32 cols,  256 neurons, 4 regions — dev/testing (~3M params)
    - 'medium': 128 cols, 384 neurons, 4 regions — single GPU (~30M params)
    - 'large':  256 cols, 512 neurons, 4 regions — one GPU (~100M params)
    - 'xl':     512 cols, 1024 neurons, 4 regions — multi-GPU (~800M params)
    """
    # Column dimensions
    n_neurons: int = 512          # neurons per column
    n_active: int = 24            # ~5% sparsity
    input_dim: int = 256          # input feature size
    context_dim: int = 256        # feedback/context feature size

    # Network topology
    n_columns: int = 256          # total columns
    n_regions: int = 4            # hierarchical regions

    # Hebbian learning
    lr_hebbian: float = 0.01
    lr_predict: float = 0.005
    lr_decode: float = 0.01
    decay: float = 0.9999
    homeostasis: float = 0.01
    feedback_strength: float = 0.3

    # Sparse encoding
    vocab_size: int = 256         # byte-level (no tokenizer)
    seq_memory: int = 256         # sequence memory slots

    # Performance
    chunk_size: int = 1           # bytes per chunk (1=byte-level, 32+=fast training)
    maintain_interval: int = 100  # norm clamping every N steps
    dtype: str = "float32"        # float32 or bfloat16

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def small(cls, **kw):
        return cls(n_columns=32, n_neurons=256, n_active=12,
                   input_dim=128, context_dim=128, n_regions=4,
                   seq_memory=128, maintain_interval=50, **kw)

    @classmethod
    def medium(cls, **kw):
        """~30M params, single GPU."""
        return cls(n_columns=128, n_neurons=384, n_active=18,
                   input_dim=192, context_dim=192, n_regions=4,
                   seq_memory=256, chunk_size=16, **kw)

    @classmethod
    def large(cls, **kw):
        """~100M params, one GPU. chunk_size=32 for fast Wikipedia training."""
        return cls(n_columns=256, n_neurons=512, n_active=24,
                   input_dim=256, context_dim=256, n_regions=4,
                   seq_memory=256, chunk_size=32, **kw)

    @classmethod
    def xl(cls, **kw):
        """~800M params, multi-GPU. chunk_size=64 for fast Wikipedia training."""
        return cls(n_columns=512, n_neurons=1024, n_active=48,
                   input_dim=512, context_dim=512, n_regions=4,
                   seq_memory=512, chunk_size=64, **kw)


class CorticalRegion(torch.nn.Module):
    """
    A region of cortical columns — ALL processed in parallel via einsum.

    Supports batched input: [B, D] processes B bytes through C columns.
    Total parallelism per forward: B * C.
    """

    def __init__(self, n_cols: int, config: CortexConfig):
        super().__init__()
        self.n_cols = n_cols
        self.config = config
        N = config.n_neurons
        D = config.input_dim
        D_ctx = config.context_dim

        # Xavier-scaled initialization
        ff_scale = math.sqrt(2.0 / (D + N))
        ctx_scale = math.sqrt(2.0 / (D_ctx + N))
        pred_scale = math.sqrt(2.0 / (N + D))

        self.W_ff = torch.nn.Parameter(torch.randn(n_cols, D, N) * ff_scale)
        self.W_ctx = torch.nn.Parameter(torch.randn(n_cols, D_ctx, N) * ctx_scale)
        self.W_pred = torch.nn.Parameter(torch.randn(n_cols, N, D) * pred_scale)
        self.bias = torch.nn.Parameter(torch.zeros(n_cols, N))
        self.register_buffer('avg_activity', torch.ones(n_cols, N) * 0.05)

    def forward(self, x_input: torch.Tensor, x_context: Optional[torch.Tensor] = None):
        """
        Forward pass through all columns.

        Args:
            x_input: [B, D] or [D]
            x_context: [B, D_ctx] or [D_ctx] or None

        Returns:
            activations, predictions, errors
        """
        single = x_input.dim() == 1
        if single:
            x_input = x_input.unsqueeze(0)

        # Feedforward: [B, D] x [C, D, N] -> [B, C, N]
        ff_drive = torch.einsum('bd,cdn->bcn', x_input, self.W_ff)

        if x_context is not None:
            if x_context.dim() == 1:
                x_context = x_context.unsqueeze(0)
            ctx_drive = torch.einsum('bd,cdn->bcn', x_context, self.W_ctx)
            drive = ff_drive + ctx_drive * self.config.feedback_strength
        else:
            drive = ff_drive

        drive = drive + self.bias.unsqueeze(0)

        # k-Winners-Take-All (sparse activation)
        activations = self._k_winners(drive, self.config.n_active)

        # Predictions: [B, C, N] x [C, N, D] -> [B, C, D]
        predictions = torch.einsum('bcn,cnd->bcd', activations, self.W_pred)

        # Prediction error
        errors = x_input.unsqueeze(1).expand(-1, self.n_cols, -1) - predictions

        if single:
            return activations.squeeze(0), predictions.squeeze(0), errors.squeeze(0)
        return activations, predictions, errors

    def _k_winners(self, drive: torch.Tensor, k: int) -> torch.Tensor:
        """k-Winners-Take-All with homeostatic boosting."""
        boost = torch.log1p(0.05 / (self.avg_activity + 1e-6))
        if drive.dim() == 3:
            boosted = drive + boost.unsqueeze(0) * self.config.homeostasis * 100
        else:
            boosted = drive + boost * self.config.homeostasis * 100

        topk_vals, _ = torch.topk(boosted, k, dim=-1)
        threshold = topk_vals[..., -1:].detach()

        mask = (boosted >= threshold).float()
        activation = F.relu(drive) * mask
        activation = activation / (activation.sum(dim=-1, keepdim=True) + 1e-8) * k
        return activation

    @torch.no_grad()
    def learn(self, x_input: torch.Tensor, activations: torch.Tensor,
              errors: torch.Tensor, x_context: torch.Tensor = None):
        """Hebbian learning + decay. Norm clamping deferred to maintain()."""
        cfg = self.config
        batched = x_input.dim() == 2

        if batched:
            B = x_input.shape[0]
            hebbian = torch.einsum('bd,bcn->cdn', x_input, activations) / B
            pred_update = torch.einsum('bcn,bcd->cnd', activations, errors) / B
            mean_act = activations.detach().mean(dim=0)
        else:
            hebbian = x_input.unsqueeze(0).unsqueeze(-1) * activations.unsqueeze(1)
            pred_update = activations.unsqueeze(-1) * errors.unsqueeze(1)
            mean_act = activations.detach()

        self.W_ff.data.add_(hebbian, alpha=cfg.lr_hebbian)
        self.W_pred.data.add_(pred_update, alpha=cfg.lr_predict)

        # Context learning (prevents W_ctx from decaying to zero)
        if x_context is not None:
            if batched:
                ctx_hebbian = torch.einsum('bd,bcn->cdn', x_context, activations) / B
            else:
                ctx_hebbian = x_context.unsqueeze(0).unsqueeze(-1) * activations.unsqueeze(1)
            self.W_ctx.data.add_(ctx_hebbian, alpha=cfg.lr_hebbian)

        # Decay
        self.W_ff.data *= cfg.decay
        self.W_ctx.data *= cfg.decay
        self.W_pred.data *= cfg.decay

        # Homeostasis
        self.avg_activity.lerp_(mean_act, 0.01)
        target = cfg.n_active / cfg.n_neurons
        self.bias.data += cfg.homeostasis * (target - self.avg_activity)

    @torch.no_grad()
    def maintain(self):
        """Norm clamping to prevent weight blow-up."""
        max_norm = 2.0
        ff_norm = self.W_ff.data.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.W_ff.data *= (max_norm / ff_norm).clamp(max=1.0)
        ctx_norm = self.W_ctx.data.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.W_ctx.data *= (max_norm / ctx_norm).clamp(max=1.0)
        pred_norm = self.W_pred.data.norm(dim=2, keepdim=True).clamp(min=1e-6)
        self.W_pred.data *= (max_norm / pred_norm).clamp(max=1.0)


class Cortex(torch.nn.Module):
    """
    Cortex AGI — the full cortical network.

    Regions of cortical columns with hierarchical processing,
    Hebbian learning, and byte-level text processing.

    Supports batched and sequential processing.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.config = config

        assert config.n_columns % config.n_regions == 0, \
            f"n_columns ({config.n_columns}) must be divisible by n_regions ({config.n_regions})"
        cols_per_region = config.n_columns // config.n_regions
        self.cols_per_region = cols_per_region

        self.regions = torch.nn.ModuleList([
            CorticalRegion(cols_per_region, config)
            for _ in range(config.n_regions)
        ])

        # Byte embedding
        self.byte_embed = torch.nn.Parameter(
            torch.randn(config.vocab_size, config.input_dim) * 0.1
        )

        # Sequence memory + incremental sum
        self.register_buffer('memory', torch.zeros(config.seq_memory, config.n_neurons))
        self.register_buffer('mem_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_mem_sum', torch.zeros(config.n_neurons))

        # Output decoder
        decode_scale = math.sqrt(2.0 / (config.n_neurons + config.vocab_size))
        self.W_decode = torch.nn.Parameter(
            torch.randn(config.n_neurons, config.vocab_size) * decode_scale
        )

        # Inter-region projections
        up_scale = math.sqrt(2.0 / (config.n_neurons + config.input_dim))
        down_scale = math.sqrt(2.0 / (config.n_neurons + config.context_dim))
        self.region_up = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(config.n_neurons, config.input_dim) * up_scale)
            for _ in range(config.n_regions - 1)
        ])
        self.region_down = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(config.n_neurons, config.context_dim) * down_scale)
            for _ in range(config.n_regions - 1)
        ])

        self.step_count = 0
        self._steps_since_maintain = 0
        self._step_callback = None
        self._step_callback_interval = 0
        self._steps_since_callback = 0

    def set_step_callback(self, callback, every: int = 500):
        """Set a callback(cortex) to fire every N steps (e.g., for checkpointing)."""
        self._step_callback = callback
        self._step_callback_interval = every
        self._steps_since_callback = 0

    def encode_text(self, text: str) -> torch.Tensor:
        """Byte-level text encoding — no tokenizer needed."""
        raw = list(text.encode('utf-8'))
        indices = torch.tensor(raw, dtype=torch.long, device=self.byte_embed.device)
        return self.byte_embed[indices]

    def process_batch(self, byte_embeddings: torch.Tensor) -> tuple:
        """Process B bytes through all regions in parallel."""
        cfg = self.config
        B = byte_embeddings.shape[0]

        ctx = self._mem_sum[:cfg.context_dim] * (1.0 / cfg.seq_memory)
        ctx = ctx.unsqueeze(0).expand(B, -1)

        current_input = byte_embeddings
        for region_idx, region in enumerate(self.regions):
            acts, preds, errs = region(current_input, ctx)
            region.learn(current_input, acts, errs, x_context=ctx)
            region_act = acts.mean(dim=1)

            if region_idx < cfg.n_regions - 1:
                current_input = region_act @ self.region_up[region_idx]
                ctx = region_act @ self.region_down[region_idx]

        top_act = region_act
        # Update memory with _mem_sum tracking (was missing before)
        ptr = self.mem_ptr.item()
        batch_mean = top_act.detach().mean(dim=0)
        old_val = self.memory[ptr]
        self._mem_sum += batch_mean - old_val
        self.memory[ptr] = batch_mean
        self.mem_ptr = (self.mem_ptr + 1) % cfg.seq_memory

        logits = top_act @ self.W_decode
        self.step_count += B

        self._steps_since_maintain += B
        if self._steps_since_maintain >= cfg.maintain_interval:
            for r in self.regions:
                r.maintain()
            self._steps_since_maintain = 0

        self._fire_callback(B)
        return top_act, logits

    def process_byte(self, byte_embedding: torch.Tensor, learning: bool = True) -> tuple:
        """Process one byte sequentially (maintains temporal context)."""
        cfg = self.config
        ctx = self._mem_sum[:cfg.context_dim] * (1.0 / cfg.seq_memory)
        current_input = byte_embedding
        top_act = None

        for region_idx, region in enumerate(self.regions):
            acts, preds, errs = region(current_input, ctx)
            if learning:
                region.learn(current_input, acts, errs, x_context=ctx)
            region_act = acts.mean(dim=0)

            if region_idx < cfg.n_regions - 1:
                current_input = region_act @ self.region_up[region_idx]
                ctx = region_act @ self.region_down[region_idx]
            else:
                top_act = region_act

        # Incremental memory update
        ptr = self.mem_ptr.item()
        old_val = self.memory[ptr]
        new_val = top_act.detach()
        self._mem_sum += new_val - old_val
        self.memory[ptr] = new_val
        self.mem_ptr = (self.mem_ptr + 1) % cfg.seq_memory

        logits = top_act @ self.W_decode
        self.step_count += 1

        if learning:
            self._steps_since_maintain += 1
            if self._steps_since_maintain >= cfg.maintain_interval:
                for r in self.regions:
                    r.maintain()
                self._steps_since_maintain = 0
            self._fire_callback(1)

        return top_act, logits

    def _fire_callback(self, n_steps: int):
        """Fire step callback if interval reached."""
        if self._step_callback and self._step_callback_interval > 0:
            self._steps_since_callback += n_steps
            if self._steps_since_callback >= self._step_callback_interval:
                self._step_callback(self)
                self._steps_since_callback = 0

    @torch.no_grad()
    def _clamp_decoder(self):
        max_norm = 2.0
        norms = self.W_decode.data.norm(dim=0, keepdim=True).clamp(min=1e-6)
        self.W_decode.data *= (max_norm / norms).clamp(max=1.0)

    @torch.no_grad()
    def learn_decoder_batch(self, activations: torch.Tensor, next_byte_indices: torch.Tensor):
        cfg = self.config
        B = activations.shape[0]
        targets = torch.zeros(B, cfg.vocab_size, device=activations.device)
        targets.scatter_(1, next_byte_indices.unsqueeze(1), 1.0)
        update = torch.einsum('bn,bv->nv', activations, targets) / B
        self.W_decode.data.add_(update, alpha=cfg.lr_decode)
        self._clamp_decoder()

    @torch.no_grad()
    def learn_decoder(self, activation: torch.Tensor, next_byte_idx: int):
        cfg = self.config
        target = torch.zeros(cfg.vocab_size, device=activation.device)
        target[next_byte_idx] = 1.0
        update = activation.unsqueeze(1) * target.unsqueeze(0)
        self.W_decode.data.add_(update, alpha=cfg.lr_decode)
        self._clamp_decoder()

    def feed_text(self, text: str, batch_size: int = 1, verbose: bool = False):
        """Feed text for learning. batch_size=1 for quality, >1 for speed."""
        if batch_size <= 1:
            return self._feed_sequential(text, verbose)
        return self._feed_batched(text, batch_size, verbose)

    def _feed_sequential(self, text: str, verbose: bool = False):
        raw = list(text.encode('utf-8'))
        if len(raw) < 2:
            return 0.0
        embeddings = self.encode_text(text)
        total_correct = 0
        total = 0
        prev_activation = None
        prev_logits = None

        for i, emb in enumerate(embeddings):
            if prev_activation is not None:
                self.learn_decoder(prev_activation, raw[i])
            if prev_logits is not None:
                if torch.argmax(prev_logits).item() == raw[i]:
                    total_correct += 1
                total += 1
            activation, logits = self.process_byte(emb)
            prev_activation = activation.detach()
            prev_logits = logits
            if verbose and i % 1000 == 0 and total > 0:
                print(f"    byte {i:,}/{len(embeddings):,} | acc: {total_correct/total*100:.1f}%")

        if self._steps_since_maintain > 0:
            for r in self.regions:
                r.maintain()
            self._steps_since_maintain = 0
        return total_correct / max(total, 1) * 100

    def _feed_batched(self, text: str, batch_size: int, verbose: bool = False):
        raw = list(text.encode('utf-8'))
        if len(raw) < 2:
            return 0.0
        embeddings = self.encode_text(text)
        dev = self.byte_embed.device
        total_correct = 0
        total = 0
        cs = self.config.chunk_size

        if cs > 1 and len(raw) >= cs * 2:
            # Chunk encoding: average cs byte embeddings into one "percept"
            # Like how sensory cortex processes receptive fields, not raw stimuli
            n_chunks = len(raw) // cs
            usable = n_chunks * cs
            chunks = embeddings[:usable].view(n_chunks, cs, -1).mean(dim=1)

            for start in range(0, n_chunks - 1, batch_size):
                end = min(start + batch_size, n_chunks - 1)
                batch_emb = chunks[start:end]
                # Target: first byte of the next chunk
                batch_targets = [raw[(start + j + 1) * cs] for j in range(end - start)]
                activations, logits = self.process_batch(batch_emb)
                target_tensor = torch.tensor(batch_targets, dtype=torch.long, device=dev)
                self.learn_decoder_batch(activations.detach(), target_tensor)
                predicted = torch.argmax(logits, dim=-1)
                total_correct += (predicted == target_tensor).sum().item()
                total += len(batch_targets)
        else:
            # Byte-level (original path, for small texts or chunk_size=1)
            for start in range(0, len(raw) - 1, batch_size):
                end = min(start + batch_size, len(raw) - 1)
                batch_emb = embeddings[start:end]
                batch_targets = raw[start + 1:end + 1]
                activations, logits = self.process_batch(batch_emb)
                target_tensor = torch.tensor(batch_targets, dtype=torch.long, device=dev)
                self.learn_decoder_batch(activations.detach(), target_tensor)
                predicted = torch.argmax(logits, dim=-1)
                total_correct += (predicted == target_tensor).sum().item()
                total += len(batch_targets)

                if verbose and total > 0 and start % (batch_size * 50) == 0:
                    print(f"    byte {start:,}/{len(raw):,} | acc: {total_correct/total*100:.1f}%")

        if self._steps_since_maintain > 0:
            for r in self.regions:
                r.maintain()
            self._steps_since_maintain = 0
        return total_correct / max(total, 1) * 100

    def generate(self, prompt: str, max_bytes: int = 200, temperature: float = 1.0) -> str:
        """Generate text — no learning during generation."""
        if not prompt:
            return ""

        embeddings = self.encode_text(prompt)
        for emb in embeddings:
            activation, logits = self.process_byte(emb, learning=False)

        generated = []
        for _ in range(max_bytes):
            if temperature <= 0:
                next_byte = torch.argmax(logits).item()
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()
            if next_byte == 0:
                break
            generated.append(next_byte)
            next_emb = self.byte_embed[next_byte]
            activation, logits = self.process_byte(next_emb, learning=False)

        return bytes(generated).decode('utf-8', errors='replace')

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def stats(self):
        cfg = self.config
        return {
            'columns': cfg.n_columns,
            'regions': cfg.n_regions,
            'neurons_per_column': cfg.n_neurons,
            'total_neurons': cfg.n_columns * cfg.n_neurons,
            'total_synapses': self.num_parameters(),
            'active_pct': f"{cfg.n_active / cfg.n_neurons * 100:.1f}%",
            'steps': self.step_count,
        }

    def save_checkpoint(self, path: str, extra: dict = None):
        """Save model checkpoint."""
        data = {
            'config': self.config.__dict__,
            'state_dict': self.state_dict(),
            'steps': self.step_count,
        }
        if extra:
            data.update(extra)
        torch.save(data, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = None) -> 'Cortex':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
        cfg_dict = checkpoint['config']
        cfg_dict['device'] = device if device else 'cpu'
        config = CortexConfig(**{k: v for k, v in cfg_dict.items()
                                 if k in CortexConfig.__dataclass_fields__})
        cortex = cls(config).to(config.device)
        cortex.load_state_dict(checkpoint['state_dict'])
        cortex.step_count = checkpoint.get('steps', 0)
        return cortex

    @torch.no_grad()
    def sync_weights(self):
        """Average all weights across GPUs for distributed Hebbian learning.

        Unlike gradient-based DDP, Hebbian learning modifies weights directly.
        This averages parameters across GPUs to keep models in sync.
        Memory buffers are excluded (each GPU has its own temporal context).
        """
        import torch.distributed as dist
        for param in self.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
        # Sync homeostasis buffers, but NOT per-GPU memory state
        skip = {'memory', '_mem_sum', 'mem_ptr'}
        for name, buf in self.named_buffers():
            short_name = name.split('.')[-1]
            if buf.is_floating_point() and short_name not in skip:
                dist.all_reduce(buf.data, op=dist.ReduceOp.AVG)

    def reset_memory(self):
        """Reset sequence memory buffer (call between unrelated documents)."""
        self.memory.zero_()
        self._mem_sum.zero_()
        self.mem_ptr.zero_()


if __name__ == "__main__":
    print("=" * 50)
    print("  Cortex AGI — Brain-Inspired Architecture")
    print("=" * 50)

    config = CortexConfig.small()
    cortex = Cortex(config).to(config.device)

    stats = cortex.stats()
    print(f"\n  Config: small")
    print(f"  Columns: {stats['columns']} ({config.n_regions} regions)")
    print(f"  Neurons: {stats['total_neurons']:,}")
    print(f"  Synapses: {stats['total_synapses']:,}")
    print(f"  Sparsity: {stats['active_pct']}")
    print(f"  Device: {config.device}")

    import time
    sample = "The brain is not a computer. It is something far stranger and more beautiful."

    t0 = time.time()
    acc1 = cortex.feed_text(sample)
    t1 = time.time()
    print(f"\n  Sequential pass: {acc1:.1f}% | {t1-t0:.3f}s")

    acc2 = cortex.feed_text(sample)
    print(f"  Pass 2: {acc2:.1f}%")

    for _ in range(8):
        acc = cortex.feed_text(sample)
    print(f"  After 10 passes: {acc:.1f}%")

    output = cortex.generate("The brain", max_bytes=100, temperature=1.0)
    safe = output.encode('ascii', errors='replace').decode('ascii')
    print(f"\n  Generate: {safe}")

    t0 = time.time()
    acc_batch = cortex.feed_text(sample * 10, batch_size=32)
    t1 = time.time()
    bps = len((sample * 10).encode('utf-8')) / (t1 - t0)
    print(f"\n  Batched: {acc_batch:.1f}% | {bps:.0f} B/s")

    print(f"\n  Steps: {cortex.step_count:,}")
    print("=" * 50)
