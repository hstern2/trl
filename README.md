# trl

Domain-agnostic training for autoregressive token-sequence transformers, with
multi-objective REINFORCE fine-tuning.

`trl` operates only on token IDs and user-supplied objective plug-ins. It has
no AMSR, chemistry, or molecule dependency; application packages such as
`mtrl` provide domain-specific decoding, validation, and scoring.
The Python library is the primary interface; the CLI is a thin wrapper around
its generic indexing, training, sampling, and RL functions.

## What is implemented

- Decoder-only Transformer with RoPE, SwiGLU, RMSNorm, tied embeddings, and
  causal scaled-dot-product attention.
- Bounded-memory JSONL indexing into compact token IDs, 64-bit row offsets,
  and 16-bit sequence lengths. Indices are built atomically once and then
  memory-mapped by every data-loading worker.
- Deterministic bounded shuffling, length bucketing, exact DDP sharding, and
  complete row coverage, including the final partial batch.
- Token-weighted gradient accumulation, FP16 plus dynamic loss scaling on V100,
  BF16 on Ampere and newer GPUs, clipping, label smoothing, and z-loss.
- Explicit frozen validation data with sharded, globally reduced loss.
- Atomic, exact-resume checkpoints containing model, optimizer, scheduler,
  scaler, data cursor, early-stopping state, and per-rank RNG state.
- Checkpoint-compatible vocabulary expansion for continued pretraining.

## Installation

```bash
uv sync --extra dev
```

`torch.compile` is optional. Triton compilation needs the Python development
headers; for example, `python3.12-dev` on Ubuntu. If `Python.h` or Triton is
unavailable, training reports that fact and falls back to eager mode before the
first model step.

## Index behavior

The first launch builds `train.*` and `validation.*` under `--index-dir` on
rank 0 while other ranks wait. Later launches validate source path, size,
modification time, vocabulary hash, and binary file sizes, then reuse the
index. Pass `--rebuild-index` to regenerate it deliberately.

Every JSONL row must be a non-empty JSON list of known token strings. The
indexer raises on blank, malformed, unknown-token, or excessively long rows;
training raises if an indexed row exceeds `--max-seq`. Rows are never silently
filtered, truncated, resplit, or dropped.

## Generic pretraining

```bash
uv run torchrun --standalone --nproc_per_node=2 -m trl pretrain train.jsonl \
  --val-data validation.jsonl \
  --vocab vocab.json \
  --index-dir .trl-index
```

For a corpus whose first index build can exceed the distributed process-group
timeout, build the reusable indices before launching `torchrun`:

```bash
uv run python -m trl index train.jsonl \
  --val-data validation.jsonl \
  --vocab vocab.json \
  --index-dir .trl-index
```

The subsequent `pretrain` command validates and reuses those files without
rescanning the corpus.

If the vocabulary path does not exist, rank 0 builds it from all supplied
training and validation files. Without an initialization checkpoint, the CLI
selects a model from corpus statistics. `--max-steps` overrides the default
`--epochs 1` budget. Use `python -m trl pretrain --help` for all options.

Use `--val-at-start` to establish the warm-start loss before optimization.
`--shadow-val-data` adds a separately reported validation view that never
affects best-checkpoint selection or early stopping; control its cadence with
`--shadow-val-every`.

## Sampling

```bash
uv run trl sample checkpoints/best.pt -n 5000 --temperature 0.8
```

The vocabulary is embedded in new checkpoints and is loaded automatically.
PAD and BOS are never sampled as generated tokens.

## RL fine-tuning

```bash
uv run torchrun --standalone --nproc_per_node=2 -m trl rl \
  checkpoints/best.pt corpus.jsonl \
  --objectives mypackage.objectives:build
```

An objectives module exposes a factory returning
`trl.objectives.base.Objectives`. Individual objectives subclass `Objective`:

```python
from trl.objectives.base import Objective


class MyObjective(Objective):
    def score_batch(self, items: list) -> list[float]:
        ...
```

## Data format

Each JSONL line is a token sequence:

```json
["token1","token2","token3"]
["tokenA","tokenB"]
```

The index adds BOS and EOS. Token ID zero is padding and is excluded from loss.

## Development checks

```bash
uv run pytest -q
uv run ruff check trl tests
uv run mypy trl
```
