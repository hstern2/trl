# trl

Domain-agnostic library for training autoregressive transformers on arbitrary token sequences, then fine-tuning with multi-objective REINFORCE.

## Architecture

- **Decoder-only transformer**: RoPE, SwiGLU FFN, RMSNorm, weight-tied LM head, depth-scaled residual init. Falls back from Flash Attention to `F.scaled_dot_product_attention` when flash-attn is unavailable.
- **Pretrain**: next-token prediction with DDP, `torch.compile`, cosine LR + warmup, bf16 autocast with fp32 master weights, label smoothing, z-loss, gradient clipping, step-budget training with deterministic hash-based val split and val-loss early stopping, wandb logging.
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
# 1. Pretrain (4x GPU). One command: scans corpora, builds vocab, picks
#    sensible model size / lr / batch / max_steps from corpus stats, and
#    trains. Multiple corpora are concatenated. Any auto-chosen value can
#    be overridden with a flag.
torchrun --nproc_per_node=4 -m trl pretrain corpus1.jsonl corpus2.jsonl

# 2. Sample (vocab is loaded from checkpoint automatically)
trl sample checkpoints/best.pt -n 5000 --temperature 0.8

# 3. RL fine-tune with external objectives
torchrun --nproc_per_node=4 -m trl rl checkpoints/best.pt corpus1.jsonl \
    --objectives mtrl.objectives:build
```

`trl pretrain` scans the corpus on startup, (re)builds `vocab.json` from
it, and picks a Chinchilla-style starting configuration (~20 tokens per parameter; `lr` from `d_model`; batch aiming
for ~100k tokens/optimizer-step; `max_seq` from the 99th-percentile length;
`max_steps` at 3× the Chinchilla target with early stopping via validation
loss as the real stop criterion). It then prints exactly what it decided
and which values came from the command line vs. auto-selection. Override
anything by passing the corresponding flag:

```bash
# Auto everything:
torchrun --nproc_per_node=1 -m trl pretrain cod.jsonl

# Keep most auto choices but force a larger model and longer budget:
torchrun --nproc_per_node=1 -m trl pretrain cod.jsonl qmugs.jsonl \
    --layers 8 --d-model 512 --max-steps 60000
```

Example stdout from a pretrain run:

```
[scan] reading 1 corpus file(s): cod.jsonl
[scan] 601,994 sequences  17,899,164 tokens  (avg=29 p50=27 p99=76 max=130)  vocab=210
[vocab] wrote vocab.json (210 tokens)
[config] resolved hyperparameters:
  layers        = 4          (auto)
  d_model       = 128        (auto)
  heads         = 4          (auto)
  d_ff          = 384        (auto)
  max_seq       = 80         (auto)
  batch_size    = 1024       (auto, per GPU × 1 = 30k tokens/step)
  lr            = 6.00e-04   (auto)
  warmup_steps  = 200        (auto)
  max_steps     = 3,000      (auto, ≈5.2 passes over train)
[model] est. params = 0.88M (Chinchilla target ≈ 18M tokens for 18M available)
```

## CLI reference

### `trl pretrain`

Pretrain a next-token transformer on one or more JSONL corpora. Launch with `torchrun --nproc_per_node=N -m trl pretrain`.

On startup, `trl pretrain` scans the corpora, builds or loads `vocab.json`, and picks sensible defaults for everything marked *auto* below from the corpus statistics (Chinchilla-style ~20 tokens per parameter). Any flag you pass overrides the corresponding auto choice; `d_ff` and `lr` are re-derived from whatever `d_model` ends up being.

| Option | Default | Description |
|--------|---------|-------------|
| `DATA` (arg) | required | One or more JSONL corpus files |
| `--vocab` | `vocab.json` | Vocab JSON output path (always rebuilt from the corpus) |
| `--layers` | *auto* | Number of transformer layers |
| `--d-model` | *auto* | Model dimension |
| `--heads` | *auto* | Number of attention heads |
| `--d-ff` | *auto* | FFN inner dim (auto ⇒ 8/3·d_model for SwiGLU) |
| `--max-seq` | *auto* | Maximum sequence length (auto = 99th percentile) |
| `--dropout` | `0.1` | Dropout rate |
| `--max-steps` | *auto* | Max optimizer steps (early stopping may end sooner) |
| `--batch-size` | *auto* | Batch size per GPU |
| `--lr` | *auto* | Learning rate |
| `--warmup-steps` | *auto* | LR warmup steps |
| `--grad-clip` | `1.0` | Gradient clipping norm |
| `--z-loss` | `1e-4` | Z-loss coefficient for logit stability (0 to disable) |
| `--val-fraction` | `0.01` | Validation holdout fraction (0 to disable) |
| `--val-every` | `500` | Evaluate on val every N steps |
| `--patience` | `10` | Early stop after N evals with no val improvement (0 to disable) |
| `--compile` / `--no-compile` | `--compile` | Use `torch.compile` |
| `--checkpoint-dir` | `checkpoints/` | Checkpoint output directory |
| `--checkpoint-every` | `5000` | Save step checkpoint every N steps |
| `--log-every` | `50` | Print progress every N steps (0 to disable) |
| `--wandb-project` | disabled | W&B project name |

### `trl sample`

Sample sequences from a trained model; writes concatenated tokens to stdout, one sequence per line.

| Option | Default | Description |
|--------|---------|-------------|
| `CHECKPOINT` (arg) | required | Model checkpoint (`.pt`) |
| `--vocab` | from checkpoint | Vocab JSON (overrides checkpoint vocab) |
| `-n`, `--n-samples` | `1000` | Number of sequences to sample |
| `--temperature` | `1.0` | Sampling temperature (higher = more random) |
| `--top-k` | `0` | Top-k filtering (0 = disabled) |

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
