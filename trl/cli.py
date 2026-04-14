import typer

app = typer.Typer(
    help="trl: token-sequence transformer + RL",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def build_vocab(
    corpus: list[str] = typer.Argument(..., help="One or more JSONL files (each line a JSON list of token strings)"),
    output: str = typer.Option("vocab.json", help="Output vocab JSON path"),
    min_freq: int = typer.Option(1, help="Minimum token frequency to include"),
) -> None:
    """Build vocabulary from one or more JSONL corpora."""
    from trl.data.vocab import Vocab

    v = Vocab.build(corpus, min_freq=min_freq)
    v.save(output)
    typer.echo(f"Saved vocab ({v.size} tokens) to {output}")


@app.command()
def suggest(
    corpus: list[str] = typer.Argument(..., help="One or more JSONL files"),
    gpus: int = typer.Option(1, help="Number of GPUs you plan to train on"),
) -> None:
    """Suggest model size / lr / batch size / epochs based on corpus stats.

    Uses Chinchilla-style scaling (~20 tokens per parameter) as a starting
    point. These are heuristics — always validate on a short run first.
    """
    import json
    import math

    n_seqs = 0
    n_tokens = 0
    vocab_set: set[str] = set()
    lens: list[int] = []
    for path in corpus:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = json.loads(line)
                n_seqs += 1
                L = len(toks) + 2  # BOS/EOS
                n_tokens += L
                lens.append(L)
                vocab_set.update(toks)

    lens.sort()
    vocab_size = len(vocab_set) + 3  # + specials
    p50 = lens[len(lens) // 2]
    p99 = lens[min(len(lens) - 1, int(0.99 * len(lens)))]
    max_len = lens[-1]

    # Chinchilla target: ~20 tokens / param. Snap to a reasonable model size.
    target_params = n_tokens / 20

    def pick_shape(target: float) -> tuple[int, int, int]:
        candidates = [
            # (layers, d_model, heads)
            (4, 128, 4),
            (4, 192, 6),
            (4, 256, 4),
            (6, 256, 4),
            (6, 384, 6),
            (6, 512, 8),
            (8, 512, 8),
            (12, 512, 8),
            (12, 768, 12),
            (16, 768, 12),
            (24, 1024, 16),
        ]
        best = candidates[0]
        best_dist = float("inf")
        for L, D, H in candidates:
            d_ff = max(64, (D * 8 // 3 + 63) // 64 * 64)
            # Rough param count: embed (tied) + per-block (attn + SwiGLU FFN).
            p = vocab_size * D + L * (4 * D * D + 3 * D * d_ff)
            dist = abs(math.log(p / max(1, target)))
            if dist < best_dist:
                best_dist = dist
                best = (L, D, H)
        return best

    layers, d_model, heads = pick_shape(target_params)
    d_ff = max(64, (d_model * 8 // 3 + 63) // 64 * 64)
    est_params = vocab_size * d_model + layers * (4 * d_model * d_model + 3 * d_model * d_ff)

    # Chinchilla again: want ~20 epochs-worth of tokens if small dataset,
    # fewer if very large. Scale epochs so total_tokens ≈ 20 × params.
    epochs = max(1, round((20 * est_params) / max(1, n_tokens)))
    # Don't recommend > 40 epochs even on tiny data — diminishing returns.
    epochs = min(epochs, 40)

    # LR: standard AdamW recipe is ~3e-4 for ~500-d models; scale as 1/sqrt(d).
    lr = 3e-4 * math.sqrt(512 / d_model)
    # Batch size: aim for ~50k-200k tokens/step across all GPUs.
    # avg_len * batch_size * gpus ≈ 100k.
    avg_len = max(1, n_tokens // max(1, n_seqs))
    per_gpu_batch = max(32, min(1024, int(100_000 / avg_len / gpus)))
    # Round to nearest multiple of 16 for GPU friendliness.
    per_gpu_batch = max(16, (per_gpu_batch // 16) * 16)

    max_seq = int(max(64, min(512, p99 // 16 * 16 + 16)))
    warmup_steps = 1000 if n_tokens < 50_000_000 else 2000

    typer.echo(
        f"Corpus stats:\n"
        f"  files:        {len(corpus)}\n"
        f"  sequences:    {n_seqs:,}\n"
        f"  tokens:       {n_tokens:,}  (avg={avg_len}, p50={p50}, p99={p99}, max={max_len})\n"
        f"  vocab size:   {vocab_size}\n"
        f"\n"
        f"Suggested config (Chinchilla ~20 tok/param):\n"
        f"  layers:       {layers}\n"
        f"  d_model:      {d_model}\n"
        f"  heads:        {heads}\n"
        f"  d_ff (auto):  {d_ff}\n"
        f"  max_seq:      {max_seq}\n"
        f"  est. params:  {est_params/1e6:.1f}M\n"
        f"  batch_size:   {per_gpu_batch}  (per GPU, ×{gpus} GPUs ≈ "
        f"{per_gpu_batch * gpus * avg_len / 1000:.0f}k tokens/step)\n"
        f"  lr:           {lr:.1e}\n"
        f"  warmup_steps: {warmup_steps}\n"
        f"  epochs (max): {epochs}   # early-stop on val_loss will usually end sooner\n"
        f"\n"
        f"Command:\n"
        f"  uv run --project ~/trl torchrun --nproc_per_node={gpus} -m trl pretrain \\\n"
        f"      {' '.join(corpus)} --vocab vocab.json \\\n"
        f"      --layers {layers} --d_model {d_model} --heads {heads} \\\n"
        f"      --max_seq {max_seq} --batch_size {per_gpu_batch} --lr {lr:.1e} \\\n"
        f"      --warmup_steps {warmup_steps} --epochs {epochs}"
    )


@app.command()
def prepare(
    corpus: str = typer.Argument(..., help="JSONL file (each line: JSON list of token strings)"),
    vocab: str = typer.Option("vocab.json", help="Vocab JSON path"),
    output: str = typer.Option("data.bin", help="Output binary path"),
) -> None:
    """Encode JSONL token sequences to memory-mapped binary for training."""
    typer.echo("prepare: not yet implemented (use JSONL directly for now)")


@app.command()
def pretrain(
    data: list[str] = typer.Argument(..., help="One or more JSONL corpus files"),
    vocab: str = typer.Option("vocab.json", help="Vocab JSON path"),
    layers: int = typer.Option(8, help="Number of transformer layers"),
    d_model: int = typer.Option(512, help="Model dimension"),
    heads: int = typer.Option(8, help="Number of attention heads"),
    d_ff: int = typer.Option(0, help="FFN inner dim (0 = 8/3*d_model for SwiGLU)"),
    max_seq: int = typer.Option(192, help="Maximum sequence length"),
    dropout: float = typer.Option(0.1, help="Dropout rate"),
    epochs: int = typer.Option(10, help="Max training epochs (early stopping may end sooner)"),
    batch_size: int = typer.Option(256, help="Batch size (per-GPU)"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    warmup_steps: int = typer.Option(2000, help="LR warmup steps"),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping norm"),
    z_loss: float = typer.Option(1e-4, help="Z-loss coefficient for logit stability (0 to disable)"),
    val_fraction: float = typer.Option(0.01, help="Validation holdout fraction (0 to disable)"),
    val_every: int = typer.Option(500, help="Evaluate on val every N steps"),
    patience: int = typer.Option(10, help="Early stop after N evals with no val improvement (0 to disable)"),
    compile_model: bool = typer.Option(True, "--compile/--no-compile", help="Use torch.compile"),
    checkpoint_dir: str = typer.Option("checkpoints/", help="Checkpoint output directory"),
    checkpoint_every: int = typer.Option(5000, help="Save step checkpoint every N steps"),
    log_every: int = typer.Option(50, help="Print progress every N steps (0 to disable)"),
    wandb_project: str | None = typer.Option(None, help="W&B project name (disabled if unset)"),
) -> None:
    """Pretrain (next-token). Launch: torchrun --nproc_per_node=N -m trl pretrain DATA [DATA ...]"""
    from trl.training.pretrain import pretrain as _pretrain

    _pretrain(
        data_path=data if len(data) > 1 else data[0],
        vocab_path=vocab,
        layers=layers,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff if d_ff > 0 else None,
        max_seq=max_seq,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        z_loss=z_loss,
        val_fraction=val_fraction,
        val_every=val_every,
        patience=patience,
        compile_model=compile_model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        wandb_project=wandb_project,
    )


@app.command()
def rl(
    checkpoint: str = typer.Argument(..., help="Pretrained checkpoint"),
    data: str = typer.Argument(..., help="Dataset .bin (for reference model)"),
    vocab: str = typer.Option(None, help="Vocab JSON (default: use vocab from checkpoint)"),
    objectives: str = typer.Option(..., help="Import path to objectives factory, e.g. mtrl.objectives:build"),
    iterations: int = typer.Option(10000, help="Number of RL iterations"),
    batch_size: int = typer.Option(512, help="Batch size (total across all GPUs)"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    kl_beta: float = typer.Option(0.05, help="KL penalty coefficient"),
    pareto_lambda: float = typer.Option(0.1, help="Pareto reward mixing weight"),
    temperature: float = typer.Option(1.0, help="Initial sampling temperature"),
    temperature_final: float = typer.Option(0.8, help="Final sampling temperature (linear anneal)"),
    replay_fraction: float = typer.Option(0.1, help="Fraction of batch from replay buffer"),
    checkpoint_dir: str = typer.Option("checkpoints_rl/", help="Checkpoint output directory"),
    wandb_project: str | None = typer.Option(None, help="W&B project name (disabled if unset)"),
) -> None:
    """RL fine-tune with Pareto REINFORCE. Launch: torchrun --nproc_per_node=N -m trl rl ..."""
    from trl.training.rl_train import rl_train

    rl_train(
        checkpoint_path=checkpoint,
        vocab_path=vocab,
        objectives_path=objectives,
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        kl_beta=kl_beta,
        pareto_lambda=pareto_lambda,
        temperature=temperature,
        temperature_final=temperature_final,
        replay_fraction=replay_fraction,
        checkpoint_dir=checkpoint_dir,
        wandb_project=wandb_project,
    )


@app.command()
def sample(
    checkpoint: str = typer.Argument(..., help="Model checkpoint (.pt)"),
    vocab: str = typer.Option(None, help="Vocab JSON (default: use vocab from checkpoint)"),
    n_samples: int = typer.Option(1000, "-n", "--n_samples", help="Number of sequences to sample"),
    temperature: float = typer.Option(1.0, help="Sampling temperature (higher = more random)"),
    top_k: int = typer.Option(0, help="Top-k filtering (0 = disabled)"),
) -> None:
    """Sample sequences from a trained model; writes concatenated tokens to stdout."""
    import sys

    import torch

    from trl.data.vocab import Vocab
    from trl.generation.sampler import sample as _sample
    from trl.model.transformer import TransformerConfig, TransformerLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if vocab is not None:
        v = Vocab.load(vocab)
    elif "vocab" in ckpt:
        v = Vocab(ckpt["vocab"])
    else:
        raise typer.BadParameter("Checkpoint has no embedded vocab; pass --vocab explicitly")

    config = TransformerConfig(**ckpt["config"])
    model = TransformerLM(config).to(device)
    state = {k.removeprefix("module."): val for k, val in ckpt["model"].items()}
    model.load_state_dict(state)

    sequences = _sample(model, n_samples, temperature=temperature, top_k=top_k, device=device)

    for seq in sequences:
        sys.stdout.write("".join(v.decode(seq)) + "\n")
