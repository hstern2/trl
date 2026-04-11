# trl

Domain-agnostic library for training autoregressive transformers on arbitrary token sequences, then fine-tuning with multi-objective REINFORCE.

## Architecture

- **Decoder-only transformer** (~25-30M params): 8 layers, 512 d_model, 8 heads, RoPE, SiLU FFN, RMSNorm, weight-tied LM head. Falls back from Flash Attention to `F.scaled_dot_product_attention` when flash-attn is unavailable.
- **Pretrain**: next-token prediction with DDP, cosine LR with warmup, bfloat16, gradient clipping, wandb logging.
- **RL fine-tune**: REINFORCE with Pareto-ranked rewards (NSGA-II + crowding distance), KL penalty against a frozen reference model, learned value baseline, and a replay buffer.
- **Sampling**: temperature, top-k, top-p with KV-cache decoding.

## Installation

```bash
uv pip install -e ".[dev]"

# With Flash Attention (CUDA only):
uv pip install -e ".[dev,flash]"
```

## Usage

```bash
# 1. Build vocabulary from JSONL (each line: JSON list of token strings)
trl build-vocab corpus.jsonl --output vocab.json

# 2. Pretrain (4x GPU)
torchrun --nproc_per_node=4 -m trl pretrain corpus.jsonl \
    --vocab vocab.json --epochs 10

# 3. Sample
trl sample checkpoints/best.pt --vocab vocab.json --n 5000

# 4. RL fine-tune with external objectives
torchrun --nproc_per_node=4 -m trl rl checkpoints/best.pt corpus.jsonl \
    --vocab vocab.json --objectives mtrl.suite:build
```

## Objectives interface

trl is domain-agnostic. To add new scoring objectives, subclass `trl.objectives.base.Objective`:

```python
from trl.objectives.base import Objective

class MyObjective(Objective):
    def score_batch(self, items: list) -> list[float]:
        ...
```

Then pass a `decode_fn` (token strings -> your domain object) and a list of objectives to `Objectives(...)`. The `--objectives` CLI flag accepts a Python import path to a factory function (e.g. `mypackage.suite:build`).

## Data format

JSONL where each line is a JSON list of token strings:
```json
["token1", "token2", "token3"]
["tokenA", "tokenB"]
```

## Structure

```
trl/
  data/        vocab, dataset, bucketed DataLoader
  model/       transformer, RoPE, attention, value head
  training/    pretrain (DDP), RL train (REINFORCE + Pareto)
  objectives/  abstract Objective, NSGA-II Pareto sorting
  generation/  sampling with KV-cache
```
