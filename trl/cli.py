import typer

app = typer.Typer(
    help="trl: token-sequence transformer + RL",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("index")
def index_corpora(
    data: list[str] = typer.Argument(..., help="One or more training JSONL files"),
    val_data: list[str] = typer.Option(
        [],
        "--val-data",
        help="Primary validation JSONL file; repeat for multiple files",
    ),
    shadow_val_data: list[str] = typer.Option(
        [],
        "--shadow-val-data",
        help="Shadow validation JSONL file; repeat for multiple files",
    ),
    vocab: str = typer.Option(..., help="Existing vocabulary JSON"),
    index_dir: str = typer.Option(".trl-index", help="Output index directory"),
    rebuild_index: bool = typer.Option(
        False,
        "--rebuild-index",
        help="Rebuild indices even when source fingerprints match",
    ),
) -> None:
    """Build or validate reusable corpus indices without initializing DDP."""
    from pathlib import Path

    from trl.data.dataset import build_index
    from trl.data.vocab import Vocab

    vocab_path = Path(vocab)
    if not vocab_path.is_file():
        raise typer.BadParameter(f"vocabulary does not exist: {vocab_path}")
    vocab_obj = Vocab.load(str(vocab_path))
    typer.echo(f"[vocab] {vocab_path} ({vocab_obj.size} tokens)")
    typer.echo(
        f"[index] {'building or validating' if rebuild_index else 'validating or reusing'} "
        f"indices in {index_dir}"
    )
    groups = (
        ("train", data),
        ("validation", val_data),
        ("shadow_validation", shadow_val_data),
    )
    for name, paths in groups:
        if not paths:
            continue
        metadata = build_index(
            paths,
            vocab_obj,
            index_dir,
            name,
            force=rebuild_index,
            progress=True,
        )
        typer.echo(
            f"[index] {name}={metadata.rows:,} sequences "
            f"{metadata.tokens:,} tokens"
        )


@app.command()
def pretrain(
    data: list[str] = typer.Argument(..., help="One or more JSONL corpus files"),
    val_data: list[str] = typer.Option(
        [],
        "--val-data",
        help="Frozen validation JSONL file; repeat for multiple files",
    ),
    shadow_val_data: list[str] = typer.Option(
        [],
        "--shadow-val-data",
        help="Secondary validation JSONL file; reported separately from model selection",
    ),
    vocab: str = typer.Option(
        "vocab.json",
        help="Existing vocabulary, or output path when one must be built",
    ),
    index_dir: str = typer.Option(
        ".trl-index",
        help="Directory for reusable memory-mapped corpus indices",
    ),
    rebuild_index: bool = typer.Option(
        False,
        "--rebuild-index",
        help="Rebuild indices even when their source fingerprints match",
    ),
    layers: int = typer.Option(
        0,
        help="Number of transformer layers (0 = auto or checkpoint)",
    ),
    d_model: int = typer.Option(0, help="Model dimension (0 = auto)"),
    heads: int = typer.Option(0, help="Number of attention heads (0 = auto)"),
    d_ff: int = typer.Option(0, help="FFN inner dim (0 = auto 8/3*d_model for SwiGLU)"),
    max_seq: int = typer.Option(
        0,
        help="Maximum sequence length (0 = cover corpus and checkpoint)",
    ),
    tokens_per_param: float = typer.Option(
        10.0,
        help="Sizing ratio for automatic model selection",
    ),
    dropout: float = typer.Option(0.1, help="Dropout rate"),
    epochs: float = typer.Option(1.0, help="Training passes when --max-steps is omitted"),
    max_steps: int = typer.Option(0, help="Optimizer-step limit (0 = derive from epochs)"),
    batch_size: int = typer.Option(0, help="Microbatch sequences per GPU (0 = auto)"),
    grad_accum_steps: int = typer.Option(
        0,
        help="Microbatches per optimizer step (0 = target global batch)",
    ),
    global_batch_sequences: int = typer.Option(
        1024,
        help="Target global sequence batch when accumulation is automatic",
    ),
    lr: float = typer.Option(0.0, help="Learning rate (0 = auto)"),
    warmup_steps: int = typer.Option(-1, help="LR warmup steps (-1 = auto)"),
    weight_decay: float = typer.Option(0.1, help="AdamW weight decay"),
    label_smoothing: float = typer.Option(0.1, help="Cross-entropy label smoothing"),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping norm"),
    z_loss: float = typer.Option(
        1e-4,
        help="Z-loss coefficient for logit stability (0 to disable)",
    ),
    val_every: int = typer.Option(10_000, help="Run full validation every N optimizer steps"),
    shadow_val_every: int = typer.Option(
        0,
        help="Run shadow validation every N steps (0 = final only)",
    ),
    val_at_start: bool = typer.Option(
        False,
        "--val-at-start",
        help="Evaluate validation sets at step 0 and establish the warm-start baseline",
    ),
    patience: int = typer.Option(
        10,
        help="Early stop after N evals with no val improvement (0 to disable)",
    ),
    precision: str = typer.Option(
        "auto",
        help="auto, fp16, bf16, or fp32 (auto uses FP16 on V100)",
    ),
    compile_model: bool = typer.Option(True, "--compile/--no-compile", help="Use torch.compile"),
    checkpoint_dir: str = typer.Option("checkpoints/", help="Checkpoint output directory"),
    checkpoint_every: int = typer.Option(5000, help="Save step checkpoint every N steps"),
    log_every: int = typer.Option(50, help="Print progress every N steps (0 to disable)"),
    num_workers: int = typer.Option(4, help="Data-loading worker processes per rank"),
    seed: int = typer.Option(0, help="Training and bounded-shuffle seed"),
    wandb_project: str | None = typer.Option(None, help="W&B project name (disabled if unset)"),
    init_checkpoint: str | None = typer.Option(
        None,
        help="Warm-start model weights and token IDs; optimizer/scheduler start fresh",
    ),
    resume: str | None = typer.Option(
        None,
        help="Resume a version-2 training checkpoint exactly",
    ),
    auto_resume: bool = typer.Option(
        False,
        "--auto-resume",
        help="Resume last.pt or the newest periodic checkpoint when present",
    ),
) -> None:
    """Pretrain on indexed JSONL without materializing the corpus in RAM."""
    import math
    import os
    from pathlib import Path

    import torch

    from trl.data.dataset import IndexMetadata, build_index
    from trl.data.vocab import Vocab
    from trl.training.auto_config import (
        CorpusStats,
        default_d_ff,
        default_lr,
        scan_corpora,
        suggest_config,
    )
    from trl.training.pretrain import pretrain as _pretrain
    from trl.training.utils import cleanup_ddp, setup_ddp
    from trl.training.warm_start import merge_checkpoint_vocab, read_checkpoint

    if auto_resume and resume:
        raise typer.BadParameter("--auto-resume and --resume are mutually exclusive")
    if auto_resume:
        checkpoint_root = Path(checkpoint_dir)
        last_checkpoint = checkpoint_root / "last.pt"
        if last_checkpoint.is_file():
            resume = str(last_checkpoint)
        else:
            periodic: list[tuple[int, Path]] = []
            for candidate in checkpoint_root.glob("step_*.pt"):
                try:
                    periodic.append((int(candidate.stem.removeprefix("step_")), candidate))
                except ValueError:
                    continue
            if periodic:
                resume = str(max(periodic)[1])
        if resume is not None:
            init_checkpoint = None
            typer.echo(f"[auto-resume] selected {resume}")
    if init_checkpoint and resume:
        raise typer.BadParameter("--init-checkpoint and --resume are mutually exclusive")
    if epochs <= 0:
        raise typer.BadParameter("--epochs must be positive")
    if global_batch_sequences <= 0:
        raise typer.BadParameter("--global-batch-sequences must be positive")

    setup_ddp()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    distributed = torch.distributed.is_initialized()

    def barrier() -> None:
        if not distributed:
            return
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        else:
            torch.distributed.barrier()

    checkpoint_path = resume or init_checkpoint
    checkpoint_state = read_checkpoint(checkpoint_path) if checkpoint_path else None
    resume_run = checkpoint_state.get("run_config", {}) if resume and checkpoint_state else {}

    vocab_path = Path(vocab)
    if not vocab_path.exists():
        if is_main:
            typer.echo(
                f"[vocab] scanning "
                f"{len(data) + len(val_data) + len(shadow_val_data)} corpus file(s)"
            )
            _, built_vocab = scan_corpora([*data, *val_data, *shadow_val_data])
            if checkpoint_state is not None:
                vocab_obj, _ = merge_checkpoint_vocab(checkpoint_state, built_vocab)
            else:
                vocab_obj = built_vocab
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            vocab_obj.save(str(vocab_path))
        barrier()
    corpus_vocab = Vocab.load(str(vocab_path))
    if resume and checkpoint_state is not None:
        vocab_obj = Vocab(dict(checkpoint_state["vocab"]))
        if corpus_vocab.token_to_id != vocab_obj.token_to_id:
            raise typer.BadParameter(
                "--vocab does not exactly match the resume checkpoint vocabulary"
            )
        added_tokens: list[str] = []
    elif checkpoint_state is not None:
        vocab_obj, added_tokens = merge_checkpoint_vocab(checkpoint_state, corpus_vocab)
        if vocab_obj.token_to_id != corpus_vocab.token_to_id:
            raise typer.BadParameter(
                "the existing --vocab does not preserve checkpoint token IDs; "
                "provide a compatible merged vocabulary"
            )
    else:
        vocab_obj = corpus_vocab
        added_tokens = []

    if is_main:
        typer.echo(f"[vocab] {vocab_path} ({vocab_obj.size} tokens)")
        if checkpoint_state is not None and not resume:
            typer.echo(
                f"[vocab] preserved {len(checkpoint_state['vocab'])} checkpoint IDs; "
                f"new tokens={len(added_tokens)}"
            )
        typer.echo(
            f"[index] {'building or validating' if rebuild_index else 'validating or reusing'} "
            f"indices in {index_dir}"
        )

    if is_main:
        train_metadata = build_index(
            data,
            vocab_obj,
            index_dir,
            "train",
            force=rebuild_index,
            progress=True,
        )
        val_metadata = (
            build_index(
                val_data,
                vocab_obj,
                index_dir,
                "validation",
                force=rebuild_index,
                progress=True,
            )
            if val_data
            else None
        )
        shadow_val_metadata = (
            build_index(
                shadow_val_data,
                vocab_obj,
                index_dir,
                "shadow_validation",
                force=rebuild_index,
                progress=True,
            )
            if shadow_val_data
            else None
        )
    barrier()
    train_index_path = str(Path(index_dir) / "train.index.json")
    val_index_path = str(Path(index_dir) / "validation.index.json") if val_data else None
    shadow_val_index_path = (
        str(Path(index_dir) / "shadow_validation.index.json")
        if shadow_val_data
        else None
    )
    if not is_main:
        train_metadata = IndexMetadata.load(train_index_path)
        val_metadata = IndexMetadata.load(val_index_path) if val_index_path else None
        shadow_val_metadata = (
            IndexMetadata.load(shadow_val_index_path) if shadow_val_index_path else None
        )

    stats = CorpusStats(
        n_files=len(data),
        n_seqs=train_metadata.rows,
        n_tokens=train_metadata.tokens,
        avg_len=max(1, int(train_metadata.mean_length)),
        p50=train_metadata.p50_length,
        p99=train_metadata.p99_length,
        max_len=train_metadata.max_length,
        vocab_size=vocab_obj.size,
    )
    if is_main:
        typer.echo(
            f"[index] train={stats.n_seqs:,} sequences {stats.n_tokens:,} tokens "
            f"(mean={train_metadata.mean_length:.2f} p50={stats.p50} "
            f"p99={stats.p99} max={stats.max_len})"
        )
        if val_metadata is not None:
            typer.echo(
                f"[index] validation={val_metadata.rows:,} sequences {val_metadata.tokens:,} tokens"
            )
        if shadow_val_metadata is not None:
            typer.echo(
                f"[index] shadow_validation={shadow_val_metadata.rows:,} sequences "
                f"{shadow_val_metadata.tokens:,} tokens"
            )

    sug = suggest_config(stats, gpus=world_size, tokens_per_param=tokens_per_param)

    def pick(user: int, auto: int) -> tuple[int, bool]:
        return (user, False) if user else (auto, True)

    r_layers, auto_layers = pick(layers, sug["layers"])
    r_d_model, auto_d_model = pick(d_model, sug["d_model"])
    r_heads, auto_heads = pick(heads, sug["heads"])
    r_max_seq, auto_max_seq = pick(max_seq, sug["max_seq"])
    r_batch, auto_batch = pick(batch_size, sug["batch_size"])
    if d_ff:
        r_d_ff, auto_d_ff = d_ff, False
    else:
        r_d_ff, auto_d_ff = default_d_ff(r_d_model), True
    if lr > 0:
        r_lr, auto_lr = lr, False
    else:
        r_lr, auto_lr = default_lr(r_d_model), True
    checkpoint_arch = checkpoint_state["config"] if checkpoint_state is not None else None
    if checkpoint_arch is not None:
        checkpoint_values = {
            "layers": int(checkpoint_arch["n_layers"]),
            "d_model": int(checkpoint_arch["d_model"]),
            "heads": int(checkpoint_arch["n_heads"]),
            "d_ff": int(checkpoint_arch["d_ff"]),
        }
        user_values = {
            "layers": layers,
            "d_model": d_model,
            "heads": heads,
            "d_ff": d_ff,
        }
        for name, checkpoint_value in checkpoint_values.items():
            user_value = user_values[name]
            if user_value and user_value != checkpoint_value:
                raise typer.BadParameter(
                    f"--{name.replace('_', '-')}={user_value} is incompatible with "
                    f"initialization checkpoint value {checkpoint_value}"
                )
        r_layers = checkpoint_values["layers"]
        r_d_model = checkpoint_values["d_model"]
        r_heads = checkpoint_values["heads"]
        r_d_ff = checkpoint_values["d_ff"]
        auto_layers = auto_d_model = auto_heads = auto_d_ff = False
        checkpoint_max_seq = int(checkpoint_arch["max_seq_len"])
        if max_seq and max_seq < checkpoint_max_seq:
            raise typer.BadParameter(
                f"--max-seq={max_seq} cannot be smaller than checkpoint max_seq_len "
                f"{checkpoint_max_seq}"
            )
        if not max_seq:
            r_max_seq = checkpoint_max_seq if resume else max(checkpoint_max_seq, sug["max_seq"])
            auto_max_seq = True
        if lr <= 0:
            r_lr, auto_lr = 1e-4, True
        if not batch_size and not resume:
            # A conservative starting point for the 12x512 checkpoint on 16 GB GPUs.
            r_batch = min(r_batch, 64)
            auto_batch = True

    if resume and resume_run:
        if not batch_size:
            r_batch = int(resume_run["batch_size_per_rank"])
            auto_batch = True
        if not grad_accum_steps:
            grad_accum_steps = int(resume_run["grad_accum_steps"])
        if not max_steps:
            max_steps = int(resume_run["max_steps"])
        if warmup_steps < 0:
            warmup_steps = int(resume_run["warmup_steps"])
        if seed == 0:
            seed = int(resume_run["seed"])
        if lr <= 0:
            r_lr = float(resume_run["lr"])
            auto_lr = True

    if grad_accum_steps:
        r_accum = grad_accum_steps
        auto_accum = False
    else:
        r_accum = max(1, math.ceil(global_batch_sequences / (r_batch * world_size)))
        auto_accum = True
    microbatches_per_epoch = math.ceil(stats.n_seqs / (r_batch * world_size))
    steps_per_epoch = math.ceil(microbatches_per_epoch / r_accum)
    if max_steps:
        r_max_steps = max_steps
        auto_max_steps = False
    else:
        r_max_steps = max(1, math.ceil(epochs * steps_per_epoch))
        auto_max_steps = True
    if warmup_steps >= 0:
        r_warmup, auto_warmup = warmup_steps, False
    elif checkpoint_arch is not None:
        r_warmup = min(1000, max(200, r_max_steps // 100))
        auto_warmup = True
    else:
        r_warmup = min(2000, max(200, r_max_steps // 20))
        auto_warmup = True

    effective_batch = r_batch * world_size * r_accum
    passes = r_max_steps / max(1, steps_per_epoch)

    def tag(is_auto: bool) -> str:
        return "auto" if is_auto else "user"

    def arch_tag(is_auto: bool) -> str:
        return "checkpoint" if checkpoint_arch is not None else tag(is_auto)

    if is_main:
        typer.echo("[config] resolved hyperparameters:")
        typer.echo(f"  layers        = {r_layers:<10} ({arch_tag(auto_layers)})")
        typer.echo(f"  d_model       = {r_d_model:<10} ({arch_tag(auto_d_model)})")
        typer.echo(f"  heads         = {r_heads:<10} ({arch_tag(auto_heads)})")
        typer.echo(f"  d_ff          = {r_d_ff:<10} ({arch_tag(auto_d_ff)})")
        typer.echo(f"  max_seq       = {r_max_seq:<10} ({tag(auto_max_seq)})")
        typer.echo(f"  microbatch    = {r_batch:<10} ({tag(auto_batch)}, per GPU)")
        typer.echo(
            f"  accumulation  = {r_accum:<10} ({tag(auto_accum)}, "
            f"effective global sequences={effective_batch:,})"
        )
        typer.echo(f"  lr            = {r_lr:<10.2e} ({tag(auto_lr)})")
        typer.echo(f"  warmup_steps  = {r_warmup:<10} ({tag(auto_warmup)})")
        typer.echo(
            f"  max_steps     = {r_max_steps:<10} ({tag(auto_max_steps)}, "
            f"approximately {passes:.2f} passes)"
        )

    try:
        _pretrain(
            train_index=train_index_path,
            val_index=val_index_path,
            shadow_val_index=shadow_val_index_path,
            vocab=vocab_obj,
            layers=r_layers,
            d_model=r_d_model,
            heads=r_heads,
            d_ff=r_d_ff,
            max_seq=r_max_seq,
            dropout=dropout,
            max_steps=r_max_steps,
            batch_size=r_batch,
            grad_accum_steps=r_accum,
            lr=r_lr,
            warmup_steps=r_warmup,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            grad_clip=grad_clip,
            z_loss=z_loss,
            val_every=val_every,
            shadow_val_every=shadow_val_every,
            val_at_start=val_at_start,
            patience=patience,
            precision=precision,
            compile_model=compile_model,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            log_every=log_every,
            num_workers=num_workers,
            seed=seed,
            wandb_project=wandb_project,
            init_checkpoint=init_checkpoint,
            resume_checkpoint=resume,
        )
    except BaseException:
        cleanup_ddp()
        raise


@app.command()
def rl(
    checkpoint: str = typer.Argument(..., help="Pretrained checkpoint"),
    data: str = typer.Argument(..., help="Dataset .bin (for reference model)"),
    vocab: str = typer.Option(None, help="Vocab JSON (default: use vocab from checkpoint)"),
    objectives: str = typer.Option(
        ...,
        help="Import path to objectives factory, e.g. mtrl.objectives:build",
    ),
    iterations: int = typer.Option(10000, help="Number of RL iterations"),
    batch_size: int = typer.Option(512, help="Batch size (total across all GPUs)"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    kl_beta: float = typer.Option(0.05, help="KL penalty coefficient"),
    pareto_lambda: float = typer.Option(0.1, help="Pareto reward mixing weight"),
    temperature: float = typer.Option(1.0, help="Initial sampling temperature"),
    temperature_final: float = typer.Option(0.8, help="Final sampling temperature (linear anneal)"),
    replay_fraction: float = typer.Option(
        0.0, help="Reserved for future replay support; currently must be 0"
    ),
    precision: str = typer.Option("auto", help="Precision: auto, fp32, fp16, or bf16"),
    checkpoint_every: int = typer.Option(100, help="Save checkpoint every N iterations"),
    log_every: int = typer.Option(10, help="Report progress every N iterations"),
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
        precision=precision,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
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
