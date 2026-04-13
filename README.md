# trl

Domain-agnostic library for training autoregressive transformers on arbitrary token sequences, then fine-tuning with multi-objective REINFORCE.

## Architecture

- **Decoder-only transformer** (~25-30M params): 8 layers, 512 d_model, 8 heads, RoPE, SiLU FFN, RMSNorm, weight-tied LM head. Falls back from Flash Attention to `F.scaled_dot_product_attention` when flash-attn is unavailable.
- **Pretrain**: next-token prediction with DDP, cosine LR with warmup, bfloat16, gradient clipping, wandb logging.
- **RL fine-tune**: REINFORCE with Pareto-ranked rewards (NSGA-II + crowding distance), KL penalty against a frozen reference model, learned value baseline, and a replay buffer.
- **Sampling**: temperature, top-k, top-p with KV-cache decoding.

## Installation

```bash
uv sync --extra dev

# With Flash Attention (CUDA only):
uv sync --extra dev --extra flash
```

## Usage

```bash
# 1. Build vocabulary from JSONL (each line: JSON list of token strings)
trl build-vocab corpus.jsonl --output vocab.json

# 2. Pretrain (4x GPU)
torchrun --nproc_per_node=4 -m trl pretrain corpus.jsonl \
    --vocab vocab.json --epochs 10

# 3. Sample (vocab is loaded from checkpoint automatically)
trl sample checkpoints/best.pt --n 5000 --temperature 0.8

# 4. RL fine-tune with external objectives
torchrun --nproc_per_node=4 -m trl rl checkpoints/best.pt corpus.jsonl \
    --objectives mtrl.objectives:build
```

## CLI reference

### `trl build-vocab`

Build vocabulary from a JSONL corpus.

| Option | Default | Description |
|--------|---------|-------------|
| `CORPUS` (arg) | required | JSONL file, each line a JSON list of token strings |
| `--output` | `vocab.json` | Output vocab JSON path |
| `--min-freq` | `1` | Minimum token frequency to include |

### `trl pretrain`

Pretrain (next-token prediction). Launch with `torchrun --nproc_per_node=N -m trl pretrain`.

| Option | Default | Description |
|--------|---------|-------------|
| `DATA` (arg) | required | JSONL corpus file |
| `--vocab` | `vocab.json` | Vocab JSON path |
| `--layers` | `8` | Number of transformer layers |
| `--d-model` | `512` | Model dimension |
| `--heads` | `8` | Number of attention heads |
| `--max-seq` | `192` | Maximum sequence length |
| `--dropout` | `0.1` | Dropout rate |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `256` | Batch size (per-GPU) |
| `--lr` | `3e-4` | Learning rate |
| `--warmup-steps` | `2000` | LR warmup steps |
| `--grad-clip` | `1.0` | Gradient clipping norm |
| `--checkpoint-dir` | `checkpoints/` | Checkpoint output directory |
| `--checkpoint-every` | `5000` | Save checkpoint every N steps |
| `--wandb-project` | disabled | W&B project name |

### `trl sample`

Sample sequences from a trained model.

| Option | Default | Description |
|--------|---------|-------------|
| `CHECKPOINT` (arg) | required | Model checkpoint (`.pt`) |
| `--vocab` | from checkpoint | Vocab JSON (overrides checkpoint vocab) |
| `--n` | `1000` | Number of sequences to sample |
| `--temperature` | `1.0` | Sampling temperature (higher = more random) |
| `--top-k` | `0` | Top-k filtering (0 = disabled) |
| `--output` | `samples.txt` | Output file path |

### `trl rl`

RL fine-tune with Pareto REINFORCE. Launch with `torchrun --nproc_per_node=N -m trl rl`.

| Option | Default | Description |
|--------|---------|-------------|
| `CHECKPOINT` (arg) | required | Pretrained checkpoint |
| `DATA` (arg) | required | Dataset for reference model |
| `--vocab` | from checkpoint | Vocab JSON (overrides checkpoint vocab) |
| `--objectives` | required | Import path to objectives factory (e.g. `mtrl.objectives:build`) |
| `--iterations` | `10000` | Number of RL iterations |
| `--batch-size` | `512` | Batch size (total across all GPUs) |
| `--lr` | `1e-5` | Learning rate |
| `--kl-beta` | `0.05` | KL penalty coefficient |
| `--pareto-lambda` | `0.1` | Pareto reward mixing weight |
| `--temperature` | `1.0` | Initial sampling temperature |
| `--temperature-final` | `0.8` | Final sampling temperature (linear anneal) |
| `--replay-fraction` | `0.1` | Fraction of batch from replay buffer |
| `--checkpoint-dir` | `checkpoints_rl/` | Checkpoint output directory |
| `--wandb-project` | disabled | W&B project name |

## Objectives interface

trl is domain-agnostic. To add new scoring objectives, subclass `trl.objectives.base.Objective`:

```python
from trl.objectives.base import Objective

class MyObjective(Objective):
    def score_batch(self, items: list) -> list[float]:
        ...
```

Then pass a `decode_fn` (token strings -> your domain object) and a list of objectives to `Objectives(...)`. The `--objectives` CLI flag accepts a Python import path to a factory function (e.g. `mypackage.objectives:build`).

## Data format

JSONL where each line is a JSON list of token strings:
```json
["token1", "token2", "token3"]
["tokenA", "tokenB"]
```

## References

1. Williams (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696). *Machine Learning* 8:229-256.
2. Olivecrona et al. (2017). [Molecular de novo design through deep reinforcement learning](https://doi.org/10.1186/s13321-017-0235-x). *J. Cheminformatics* 9:48.
3. Popova et al. (2018). [Deep reinforcement learning for de novo drug design](https://doi.org/10.1126/sciadv.aap7885). *Science Advances* 4:eaap7885.
4. Liu et al. (2021). [DrugEx v2: de novo design of drug molecules by Pareto-based multi-objective reinforcement learning](https://doi.org/10.1186/s13321-021-00561-9). *J. Cheminformatics* 13:85.
5. Fromer & Coley (2023). [Computer-aided multi-objective optimization in small molecule discovery](https://doi.org/10.1016/j.patter.2023.100678). *Patterns* 4:100678.
6. Thomas et al. (2024). [Clustered Pareto-based reinforcement learning for molecular generation](https://doi.org/10.1016/j.neunet.2024.106282). *Neural Networks* 175:106282.
7. Guo et al. (2025). [Augmented direct preference optimization for efficient REINFORCE finetuning of chemical language models](https://arxiv.org/abs/2501.15971). *arXiv:2501.15971*.

## Structure

```
trl/
  data/        vocab, dataset, bucketed DataLoader
  model/       transformer, RoPE, attention, value head
  training/    pretrain (DDP), RL train (REINFORCE + Pareto)
  objectives/  abstract Objective, NSGA-II Pareto sorting
  generation/  sampling with KV-cache
```
