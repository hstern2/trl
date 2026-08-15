# trl

Domain-agnostic training for autoregressive token-sequence transformers, with
multi-objective REINFORCE fine-tuning.

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

## AMSR continued-pretraining run

On the two 16 GB V100 node, the measured safe configuration is 256 sequences
per GPU and two accumulation microbatches: an effective global batch of 1,024
sequences. A 512-sequence microbatch exhausts the GPU on long rows.

```bash
cd /home/ubuntu/trl

uv run torchrun --standalone --nproc_per_node=2 -m trl pretrain \
  /home/ubuntu/trl-train-3/out/corpus/train/tokens.jsonl \
  --val-data /home/ubuntu/trl-train-3/out/corpus/validation/tokens.jsonl \
  --vocab /home/ubuntu/trl-train-3/out/corpus/vocab.json \
  --index-dir /home/ubuntu/trl-train-3/index \
  --init-checkpoint /home/ubuntu/trl-train-3/checkpoints/qmugs_cod_step22000.pt \
  --batch-size 256 \
  --grad-accum-steps 2 \
  --epochs 1 \
  --lr 1e-4 \
  --warmup-steps 1000 \
  --precision fp16 \
  --no-compile \
  --val-every 10000 \
  --checkpoint-every 5000 \
  --checkpoint-dir /home/ubuntu/trl-runs/amsr-continued
```

With this corpus, that resolves to 124,541 optimizer steps and consumes all
127,529,655 training rows exactly once. The prebuilt index contains
4,958,690,368 model tokens and has a maximum row length of 98.

Resume with the same data and run settings, replacing `--init-checkpoint` with
the version-2 checkpoint:

```bash
uv run torchrun --standalone --nproc_per_node=2 -m trl pretrain \
  /home/ubuntu/trl-train-3/out/corpus/train/tokens.jsonl \
  --val-data /home/ubuntu/trl-train-3/out/corpus/validation/tokens.jsonl \
  --vocab /home/ubuntu/trl-train-3/out/corpus/vocab.json \
  --index-dir /home/ubuntu/trl-train-3/index \
  --resume /home/ubuntu/trl-runs/amsr-continued/last.pt \
  --batch-size 256 \
  --grad-accum-steps 2 \
  --lr 1e-4 \
  --warmup-steps 1000 \
  --precision fp16 \
  --no-compile \
  --val-every 10000 \
  --checkpoint-every 5000 \
  --checkpoint-dir /home/ubuntu/trl-runs/amsr-continued
```

The resume loader rejects changes to the model, vocabulary, index fingerprint,
world size, microbatch, accumulation, precision, step budget, warmup, or seed.

For the long run, install the included user-service template and inspect it
before starting:

```bash
mkdir -p ~/.config/systemd/user
cp ops/trl-amsr.service ops/trl-amsr-runtime-report.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now trl-amsr.service
journalctl --user -u trl-amsr.service -f
```

The unit includes `--auto-resume`: on a restart it selects `last.pt`, or the
highest numbered periodic checkpoint when `last.pt` does not yet exist. The
original initialization checkpoint is used only when the run directory has no
training checkpoint. The service initializes `runtime.json` without replacing
its original timestamp on automatic restarts. When training emits its final
`[done]` event, `trl-amsr-runtime-report.service` records that event's timestamp
and the exact elapsed wall-clock duration.

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

If the vocabulary path does not exist, rank 0 builds it from all supplied
training and validation files. Without an initialization checkpoint, the CLI
selects a model from corpus statistics. `--max-steps` overrides the default
`--epochs 1` budget. Use `python -m trl pretrain --help` for all options.

## Sampling

```bash
uv run trl sample checkpoints/best.pt -n 5000 --temperature 0.8
```

The vocabulary is embedded in new checkpoints and is loaded automatically.

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
