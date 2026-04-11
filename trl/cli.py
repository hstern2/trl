import typer

app = typer.Typer(help="trl: token-sequence transformer + RL")


@app.command()
def build_vocab(
    corpus: str = typer.Argument(..., help="JSONL file, each line a JSON list of token strings"),
    output: str = typer.Option("vocab.json"),
    min_freq: int = typer.Option(1),
) -> None:
    """Build vocabulary from a JSONL corpus."""
    from trl.data.vocab import Vocab

    v = Vocab.build(corpus, min_freq=min_freq)
    v.save(output)
    typer.echo(f"Saved vocab ({v.size} tokens) to {output}")


@app.command()
def prepare(
    corpus: str = typer.Argument(..., help="JSONL file (each line: JSON list of token strings)"),
    vocab: str = typer.Option("vocab.json"),
    output: str = typer.Option("data.bin"),
) -> None:
    """Encode JSONL token sequences to memory-mapped binary for training."""
    typer.echo("prepare: not yet implemented (use JSONL directly for now)")


@app.command()
def pretrain(
    data: str = typer.Argument(..., help="Prepared .bin dataset"),
    vocab: str = typer.Option("vocab.json"),
    layers: int = typer.Option(8),
    d_model: int = typer.Option(512),
    heads: int = typer.Option(8),
    max_seq: int = typer.Option(192),
    dropout: float = typer.Option(0.1),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(256, help="Per-GPU"),
    lr: float = typer.Option(3e-4),
    warmup_steps: int = typer.Option(2000),
    grad_clip: float = typer.Option(1.0),
    checkpoint_dir: str = typer.Option("checkpoints/"),
    checkpoint_every: int = typer.Option(5000),
    wandb_project: str | None = typer.Option(None),
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
    vocab: str = typer.Option("vocab.json"),
    objectives: str = typer.Option(..., help="Import path to objectives factory, e.g. mtrl.suite:build"),
    iterations: int = typer.Option(10000),
    batch_size: int = typer.Option(512, help="Total across all GPUs"),
    lr: float = typer.Option(1e-5),
    kl_beta: float = typer.Option(0.05),
    pareto_lambda: float = typer.Option(0.1),
    temperature: float = typer.Option(1.0),
    temperature_final: float = typer.Option(0.8),
    replay_fraction: float = typer.Option(0.1),
    checkpoint_dir: str = typer.Option("checkpoints_rl/"),
    wandb_project: str | None = typer.Option(None),
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
    checkpoint: str = typer.Argument(...),
    vocab: str = typer.Option("vocab.json"),
    n: int = typer.Option(1000),
    temperature: float = typer.Option(1.0),
    top_k: int = typer.Option(0),
    output: str = typer.Option("samples.txt"),
) -> None:
    """Sample sequences from a trained model."""
    import torch

    from trl.data.vocab import Vocab
    from trl.generation.sampler import sample as _sample
    from trl.model.transformer import TransformerConfig, TransformerLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = Vocab.load(vocab)

    config = TransformerConfig(**ckpt["config"])
    model = TransformerLM(config).to(device)
    model.load_state_dict(ckpt["model"])

    sequences = _sample(model, n, temperature=temperature, top_k=top_k, device=device)

    with open(output, "w") as f:
        for seq in sequences:
            tokens = v.decode(seq)
            f.write(" ".join(tokens) + "\n")

    typer.echo(f"Wrote {len(sequences)} samples to {output}")
