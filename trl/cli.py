import typer

app = typer.Typer(
    help="trl: token-sequence transformer + RL",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def pretrain(
    data: list[str] = typer.Argument(..., help="One or more JSONL corpus files"),
    vocab: str = typer.Option("vocab.json", help="Vocab JSON output path (always built from the corpus)"),
    layers: int = typer.Option(0, help="Number of transformer layers (0 = auto from corpus size)"),
    d_model: int = typer.Option(0, help="Model dimension (0 = auto)"),
    heads: int = typer.Option(0, help="Number of attention heads (0 = auto)"),
    d_ff: int = typer.Option(0, help="FFN inner dim (0 = auto 8/3*d_model for SwiGLU)"),
    max_seq: int = typer.Option(0, help="Maximum sequence length (0 = auto from p99 length)"),
    dropout: float = typer.Option(0.1, help="Dropout rate"),
    max_steps: int = typer.Option(0, help="Max optimizer steps (0 = auto; early stopping may end sooner)"),
    batch_size: int = typer.Option(0, help="Batch size per GPU (0 = auto)"),
    lr: float = typer.Option(0.0, help="Learning rate (0 = auto)"),
    warmup_steps: int = typer.Option(-1, help="LR warmup steps (-1 = auto)"),
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
    """Pretrain a next-token transformer on one or more JSONL corpora.

    Scans the corpus, (re)uses or builds a vocab, and picks sensible model
    size / lr / batch / max_steps defaults from corpus statistics (Chinchilla-
    style ~20 tokens per parameter). Any of those can be overridden with a
    flag. Launch with:

        torchrun --nproc_per_node=N -m trl pretrain DATA [DATA ...]
    """
    import os
    from pathlib import Path

    from trl.training.auto_config import (
        default_d_ff,
        default_lr,
        estimate_params,
        scan_corpora,
        suggest_config,
    )
    from trl.training.pretrain import pretrain as _pretrain

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0

    if is_main:
        typer.echo(
            f"[scan] reading {len(data)} corpus file(s): {', '.join(data)}"
        )
    stats, built_vocab = scan_corpora(data)
    if is_main:
        typer.echo(
            f"[scan] {stats.n_seqs:,} sequences  {stats.n_tokens:,} tokens  "
            f"(avg={stats.avg_len} p50={stats.p50} p99={stats.p99} max={stats.max_len})  "
            f"vocab={stats.vocab_size}"
        )

    vocab_path = Path(vocab)
    vocab_obj = built_vocab
    if is_main:
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_obj.save(str(vocab_path))
        typer.echo(f"[vocab] wrote {vocab_path} ({vocab_obj.size} tokens)")

    sug = suggest_config(stats, gpus=world_size)

    def pick(user, auto):
        return (user, False) if user else (auto, True)

    r_layers, auto_layers = pick(layers, sug["layers"])
    r_d_model, auto_d_model = pick(d_model, sug["d_model"])
    r_heads, auto_heads = pick(heads, sug["heads"])
    r_max_seq, auto_max_seq = pick(max_seq, sug["max_seq"])
    r_max_steps, auto_max_steps = pick(max_steps, sug["max_steps"])
    r_batch, auto_batch = pick(batch_size, sug["batch_size"])
    # d_ff and lr depend on d_model, so re-derive if d_model was overridden
    # but they weren't, instead of using the suggested value.
    if d_ff:
        r_d_ff, auto_d_ff = d_ff, False
    else:
        r_d_ff, auto_d_ff = default_d_ff(r_d_model), True
    if lr > 0:
        r_lr, auto_lr = lr, False
    else:
        r_lr, auto_lr = default_lr(r_d_model), True
    if warmup_steps >= 0:
        r_warmup, auto_warmup = warmup_steps, False
    else:
        r_warmup, auto_warmup = min(2000, max(200, r_max_steps // 20)), True

    est_params = estimate_params(vocab_obj.size, r_layers, r_d_model, r_d_ff)
    tokens_per_step = r_batch * world_size * stats.avg_len
    passes = r_max_steps * tokens_per_step / max(1, stats.n_tokens)
    target_tokens = int(20 * est_params)

    def tag(is_auto: bool) -> str:
        return "auto" if is_auto else "user"

    if is_main:
        typer.echo("[config] resolved hyperparameters:")
        typer.echo(f"  layers        = {r_layers:<10} ({tag(auto_layers)})")
        typer.echo(f"  d_model       = {r_d_model:<10} ({tag(auto_d_model)})")
        typer.echo(f"  heads         = {r_heads:<10} ({tag(auto_heads)})")
        typer.echo(f"  d_ff          = {r_d_ff:<10} ({tag(auto_d_ff)})")
        typer.echo(f"  max_seq       = {r_max_seq:<10} ({tag(auto_max_seq)})")
        typer.echo(f"  batch_size    = {r_batch:<10} ({tag(auto_batch)}, per GPU × {world_size} = {tokens_per_step/1000:.0f}k tokens/step)")
        typer.echo(f"  lr            = {r_lr:<10.2e} ({tag(auto_lr)})")
        typer.echo(f"  warmup_steps  = {r_warmup:<10} ({tag(auto_warmup)})")
        typer.echo(f"  max_steps     = {r_max_steps:<10} ({tag(auto_max_steps)}, ≈{passes:.1f} passes over train)")
        typer.echo(
            f"[model] est. params = {est_params/1e6:.2f}M "
            f"(Chinchilla target ≈ {target_tokens/1e6:.0f}M tokens "
            f"for {stats.n_tokens/1e6:.0f}M available)"
        )

    _pretrain(
        data_path=data if len(data) > 1 else data[0],
        vocab=vocab_obj,
        layers=r_layers,
        d_model=r_d_model,
        heads=r_heads,
        d_ff=r_d_ff,
        max_seq=r_max_seq,
        dropout=dropout,
        max_steps=r_max_steps,
        batch_size=r_batch,
        lr=r_lr,
        warmup_steps=r_warmup,
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
