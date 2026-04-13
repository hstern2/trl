import typer

app = typer.Typer(help="trl: token-sequence transformer + RL")


@app.command()
def build_vocab(
    corpus: str = typer.Argument(..., help="JSONL file, each line a JSON list of token strings"),
    output: str = typer.Option("vocab.json", help="Output vocab JSON path"),
    min_freq: int = typer.Option(1, help="Minimum token frequency to include"),
) -> None:
    """Build vocabulary from a JSONL corpus."""
    from trl.data.vocab import Vocab

    v = Vocab.build(corpus, min_freq=min_freq)
    v.save(output)
    typer.echo(f"Saved vocab ({v.size} tokens) to {output}")


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
    data: str = typer.Argument(..., help="JSONL corpus file"),
    vocab: str = typer.Option("vocab.json", help="Vocab JSON path"),
    layers: int = typer.Option(8, help="Number of transformer layers"),
    d_model: int = typer.Option(512, help="Model dimension"),
    heads: int = typer.Option(8, help="Number of attention heads"),
    max_seq: int = typer.Option(192, help="Maximum sequence length"),
    dropout: float = typer.Option(0.1, help="Dropout rate"),
    epochs: int = typer.Option(10, help="Training epochs"),
    batch_size: int = typer.Option(256, help="Batch size (per-GPU)"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    warmup_steps: int = typer.Option(2000, help="LR warmup steps"),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping norm"),
    checkpoint_dir: str = typer.Option("checkpoints/", help="Checkpoint output directory"),
    checkpoint_every: int = typer.Option(5000, help="Save checkpoint every N steps"),
    wandb_project: str | None = typer.Option(None, help="W&B project name (disabled if unset)"),
) -> None:
    """Pretrain (next-token). Launch: torchrun --nproc_per_node=N -m trl pretrain DATA"""
    from trl.training.pretrain import pretrain as _pretrain

    _pretrain(
        data_path=data,
        vocab_path=vocab,
        layers=layers,
        d_model=d_model,
        heads=heads,
        max_seq=max_seq,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
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
    n: int = typer.Option(1000, help="Number of sequences to sample"),
    temperature: float = typer.Option(1.0, help="Sampling temperature (higher = more random)"),
    top_k: int = typer.Option(0, help="Top-k filtering (0 = disabled)"),
    output: str = typer.Option("samples.txt", help="Output file path"),
) -> None:
    """Sample sequences from a trained model."""
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
    model.load_state_dict(ckpt["model"])

    sequences = _sample(model, n, temperature=temperature, top_k=top_k, device=device)

    with open(output, "w") as f:
        for seq in sequences:
            tokens = v.decode(seq)
            f.write(" ".join(tokens) + "\n")

    typer.echo(f"Wrote {len(sequences)} samples to {output}")
