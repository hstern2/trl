from __future__ import annotations

import hashlib
import json
import math
import random
import sysconfig
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from trl.data.dataset import (
    IndexedTokenDataset,
    get_indexed_dataloader,
    vocab_sha256,
)
from trl.data.vocab import PAD, Vocab
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.training.auto_config import default_d_ff
from trl.training.utils import (
    cleanup_ddp,
    get_lr_scheduler,
    is_main,
    load_training_checkpoint,
    save_checkpoint,
    setup_ddp,
)
from trl.training.warm_start import load_warm_start


def resolve_precision(requested: str, device: torch.device) -> str:
    """Resolve auto precision without trusting unreliable BF16 probes on V100."""
    choices = {"auto", "fp32", "fp16", "bf16"}
    if requested not in choices:
        raise ValueError(f"precision must be one of {sorted(choices)}")
    if device.type != "cuda":
        if requested not in ("auto", "fp32"):
            raise ValueError(f"{requested} precision requires CUDA")
        return "fp32"
    if requested != "auto":
        if requested == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("BF16 training requires an Ampere-or-newer GPU")
        return requested
    return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"


def _autocast_context(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def _index_fingerprint(dataset: IndexedTokenDataset) -> str:
    payload = json.dumps(asdict(dataset.metadata), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> tuple[float, int]:
    """Return exact distributed CE loss and token count for validation."""
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.int64, device=device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    for inputs, targets in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast_context(device, precision):
            logits, _ = model(inputs)
        loss = loss_fn(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.double()
        total_tokens += (targets != PAD).sum()

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    model.train()
    token_count = int(total_tokens.item())
    return float(total_loss.item() / max(1, token_count)), token_count


def _multiply_gradients(model: nn.Module, factor: float) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def pretrain(
    train_index: str,
    val_index: str | None,
    vocab: Vocab | str,
    shadow_val_index: str | None = None,
    layers: int = 8,
    d_model: int = 512,
    heads: int = 8,
    d_ff: int | None = None,
    max_seq: int = 192,
    dropout: float = 0.1,
    max_steps: int = 50_000,
    batch_size: int = 256,
    grad_accum_steps: int = 1,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    weight_decay: float = 0.1,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    z_loss: float = 1e-4,
    val_every: int = 10_000,
    shadow_val_every: int = 0,
    val_at_start: bool = False,
    patience: int = 10,
    precision: str = "auto",
    compile_model: bool = True,
    checkpoint_dir: str = "checkpoints/",
    checkpoint_every: int = 5000,
    log_every: int = 50,
    num_workers: int = 4,
    seed: int = 0,
    wandb_project: str | None = None,
    init_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
) -> None:
    if init_checkpoint and resume_checkpoint:
        raise ValueError("init_checkpoint and resume_checkpoint are mutually exclusive")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")

    local_rank = setup_ddp()
    distributed = dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    resolved_precision = resolve_precision(precision, device)

    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    if isinstance(vocab, str):
        vocab = Vocab.load(vocab)
    train_ds = IndexedTokenDataset(train_index)
    val_ds = IndexedTokenDataset(val_index) if val_index is not None else None
    shadow_val_ds = (
        IndexedTokenDataset(shadow_val_index) if shadow_val_index is not None else None
    )
    for name, dataset in (
        ("training", train_ds),
        ("validation", val_ds),
        ("shadow validation", shadow_val_ds),
    ):
        if dataset is None:
            continue
        if dataset.metadata.vocab_size != vocab.size:
            raise ValueError(
                f"{name} index vocabulary size {dataset.metadata.vocab_size} "
                f"does not match model vocabulary size {vocab.size}"
            )
        if dataset.metadata.vocab_sha256 != vocab_sha256(vocab):
            raise ValueError(f"{name} index vocabulary IDs do not match the model vocabulary")
        if dataset.metadata.max_length > max_seq:
            raise ValueError(
                f"{name} row length {dataset.metadata.max_length} exceeds max_seq={max_seq}; "
                "rows are never silently dropped"
            )

    if d_ff is None:
        d_ff = default_d_ff(d_model)
    config = TransformerConfig(
        vocab_size=vocab.size,
        n_layers=layers,
        d_model=d_model,
        n_heads=heads,
        d_ff=d_ff,
        max_seq_len=max_seq,
        dropout=dropout,
    )

    model: nn.Module = TransformerLM(config).to(device)
    warm_start = None
    if init_checkpoint is not None:
        assert isinstance(model, TransformerLM)
        warm_start = load_warm_start(model, init_checkpoint, vocab)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    if compile_model and device.type == "cuda":
        include_dir = Path(sysconfig.get_paths()["include"])
        if not (include_dir / "Python.h").is_file():
            if is_main():
                print(
                    f"[compile] {include_dir / 'Python.h'} is missing; using eager mode",
                    flush=True,
                )
            compile_model = False
    if compile_model and device.type == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError:
            if is_main():
                print("[compile] triton unavailable; using eager mode", flush=True)
            compile_model = False
    if compile_model and device.type == "cuda":
        model = cast(nn.Module, torch.compile(model, dynamic=True))
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=resolved_precision == "fp16",
        # The backward objective is a token sum and is normalized after DDP.
        # A small initial scale is equivalent to a large scale on a mean loss.
        init_scale=4.0,
    )

    train_loader, train_sampler = get_indexed_dataloader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if val_ds is not None:
        val_loader, val_sampler = get_indexed_dataloader(
            val_ds,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
        )
        val_sampler.set_epoch(0)
    shadow_val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if shadow_val_ds is not None:
        shadow_val_loader, shadow_val_sampler = get_indexed_dataloader(
            shadow_val_ds,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
        )
        shadow_val_sampler.set_epoch(0)

    microbatches_per_epoch = len(train_sampler)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / grad_accum_steps)
    run_config: dict[str, Any] = {
        "train_index_sha256": _index_fingerprint(train_ds),
        "val_index_sha256": _index_fingerprint(val_ds) if val_ds is not None else None,
        "batch_size_per_rank": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "world_size": world_size,
        "precision": resolved_precision,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "grad_clip": grad_clip,
        "z_loss": z_loss,
        "val_every": val_every,
        "patience": patience,
        "compile_model": compile_model,
    }
    if shadow_val_ds is not None:
        run_config["shadow_val_index_sha256"] = _index_fingerprint(shadow_val_ds)
        run_config["shadow_val_every"] = shadow_val_every
    if val_at_start:
        run_config["val_at_start"] = True

    step = 0
    epoch = 0
    batch_in_epoch = 0
    best_val = float("inf")
    evals_without_improve = 0
    global_tokens_seen = 0
    global_sequences_seen = 0
    last_val_step = -1
    last_shadow_val_step = -1
    if resume_checkpoint is not None:
        checkpoint = load_training_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
        )
        if checkpoint.get("vocab") != vocab.token_to_id:
            raise ValueError("resume checkpoint vocabulary does not match this run")
        if checkpoint.get("config") != asdict(config):
            raise ValueError("resume checkpoint model configuration does not match this run")
        if checkpoint.get("run_config") != run_config:
            raise ValueError("resume checkpoint run configuration does not match this run")
        state = checkpoint["training_state"]
        step = int(checkpoint["step"])
        epoch = int(state["epoch"])
        batch_in_epoch = int(state["batch_in_epoch"])
        best_val = float(state["best_val"])
        evals_without_improve = int(state["evals_without_improve"])
        global_tokens_seen = int(state.get("global_tokens_seen", 0))
        global_sequences_seen = int(state.get("global_sequences_seen", 0))
        last_val_step = int(state.get("last_val_step", -1))
        last_shadow_val_step = int(state.get("last_shadow_val_step", -1))

    def training_state() -> dict[str, Any]:
        return {
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "best_val": best_val,
            "evals_without_improve": evals_without_improve,
            "global_tokens_seen": global_tokens_seen,
            "global_sequences_seen": global_sequences_seen,
            "last_val_step": last_val_step,
            "last_shadow_val_step": last_shadow_val_step,
        }

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if is_main():
        print(
            f"[setup] params={n_params / 1e6:.2f}M vocab={vocab.size} "
            f"layers={layers} d_model={d_model} d_ff={d_ff} heads={heads} "
            f"max_seq={max_seq}",
            flush=True,
        )
        if warm_start is not None:
            print(
                f"[init] checkpoint step={warm_start.checkpoint_step:,} "
                f"vocab={warm_start.old_vocab_size}->{warm_start.new_vocab_size}; "
                "optimizer and schedule start fresh",
                flush=True,
            )
        if resume_checkpoint is not None:
            print(
                f"[resume] {resume_checkpoint} at step={step:,} "
                f"epoch={epoch + 1} batch={batch_in_epoch:,}",
                flush=True,
            )
        effective_batch = batch_size * world_size * grad_accum_steps
        print(
            f"[setup] train={len(train_ds):,} val={len(val_ds) if val_ds else 0:,} "
            f"shadow_val={len(shadow_val_ds) if shadow_val_ds else 0:,} "
            f"microbatch={batch_size}×{world_size} accumulate={grad_accum_steps} "
            f"effective_sequences={effective_batch:,} precision={resolved_precision} "
            f"workers={num_workers} compile={compile_model}",
            flush=True,
        )
        print(
            f"[setup] microbatches/epoch={microbatches_per_epoch:,} "
            f"optimizer_steps/epoch={optimizer_steps_per_epoch:,} "
            f"max_steps={max_steps:,} warmup={warmup_steps:,} lr={lr:g}",
            flush=True,
        )

    wandb_run = None
    if wandb_project and is_main():
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            config={**asdict(config), **run_config},
            resume="allow",
        )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=PAD,
        label_smoothing=label_smoothing,
        reduction="sum",
    )

    def save_best_checkpoint() -> None:
        save_checkpoint(
            model,
            optimizer,
            step,
            asdict(config),
            str(Path(checkpoint_dir) / "best.pt"),
            vocab=vocab.token_to_id,
            scheduler=scheduler,
            scaler=scaler,
            training_state=training_state(),
            run_config=run_config,
        )

    def evaluate_primary(label: str, *, baseline: bool = False) -> bool:
        nonlocal best_val, evals_without_improve, last_val_step
        assert val_loader is not None
        val_started = time.perf_counter()
        val_loss, val_tokens = _evaluate(model, val_loader, device, resolved_precision)
        last_val_step = step
        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            evals_without_improve = 0
        elif not baseline:
            evals_without_improve += 1
        if is_main():
            marker = " *baseline*" if baseline else " *new best*" if improved else ""
            print(
                f"[{label}] step {step:,} loss={val_loss:.4f} "
                f"tokens={val_tokens:,} time={time.perf_counter() - val_started:.1f}s"
                f"{marker}",
                flush=True,
            )
            if wandb_run:
                wandb_run.log({"val/loss": val_loss}, step=step)
        return improved

    def evaluate_shadow(label: str) -> None:
        nonlocal last_shadow_val_step
        assert shadow_val_loader is not None
        val_started = time.perf_counter()
        val_loss, val_tokens = _evaluate(
            model,
            shadow_val_loader,
            device,
            resolved_precision,
        )
        last_shadow_val_step = step
        if is_main():
            print(
                f"[{label}] step {step:,} loss={val_loss:.4f} "
                f"tokens={val_tokens:,} time={time.perf_counter() - val_started:.1f}s",
                flush=True,
            )
            if wandb_run:
                wandb_run.log({"shadow_val/loss": val_loss}, step=step)

    if val_at_start and step == 0:
        baseline_established = False
        if val_loader is not None:
            baseline_established = evaluate_primary("val-start", baseline=True)
        if shadow_val_loader is not None:
            evaluate_shadow("shadow-val-start")
        if baseline_established:
            save_best_checkpoint()

    stop = False
    window_started = time.perf_counter()
    window_ce = 0.0
    window_tokens = 0
    window_sequences = 0

    while step < max_steps and not stop:
        train_sampler.set_epoch(epoch, start_batch=batch_in_epoch)
        if is_main():
            print(
                f"[epoch {epoch + 1}] start at optimizer step {step:,}/{max_steps:,} "
                f"microbatch {batch_in_epoch:,}/{microbatches_per_epoch:,}",
                flush=True,
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_tokens = 0
        accumulation_sequences = 0
        accumulation_ce = 0.0
        accumulation_start_batch = batch_in_epoch
        exhausted_epoch = True
        retry_after_overflow = False

        for absolute_batch, (inputs, targets) in enumerate(
            train_loader,
            start=batch_in_epoch,
        ):
            if step >= max_steps or stop:
                exhausted_epoch = False
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            local_tokens = int((targets != PAD).sum().item())
            if accumulation_tokens == 0:
                accumulation_start_batch = absolute_batch
            is_boundary = (
                absolute_batch + 1
            ) % grad_accum_steps == 0 or absolute_batch + 1 == microbatches_per_epoch
            sync_context = (
                nullcontext() if is_boundary or not isinstance(model, DDP) else model.no_sync()
            )

            with sync_context:
                with _autocast_context(device, resolved_precision):
                    logits, _ = model(inputs)
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_targets = targets.reshape(-1)
                    ce_sum = loss_fn(flat_logits, flat_targets)
                    if z_loss > 0:
                        valid = flat_targets != PAD
                        log_partition = torch.logsumexp(flat_logits.float()[valid], dim=-1)
                        loss_sum = ce_sum + z_loss * (log_partition * log_partition).sum()
                    else:
                        loss_sum = ce_sum
                scaler.scale(loss_sum).backward()

            accumulation_tokens += local_tokens
            accumulation_sequences += inputs.size(0)
            accumulation_ce += float(ce_sum.detach().item())
            batch_in_epoch = absolute_batch + 1
            if not is_boundary:
                continue

            totals = torch.tensor(
                [accumulation_ce, float(accumulation_tokens), float(accumulation_sequences)],
                dtype=torch.float64,
                device=device,
            )
            if distributed:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            global_ce = float(totals[0].item())
            global_tokens = int(totals[1].item())
            global_sequences = int(totals[2].item())

            scaler.unscale_(optimizer)
            _multiply_gradients(model, world_size / max(1, global_tokens))
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)
            accumulation_tokens = 0
            accumulation_sequences = 0
            accumulation_ce = 0.0

            if skipped:
                batch_in_epoch = accumulation_start_batch
                retry_after_overflow = True
                if is_main():
                    print(
                        f"[amp] overflow at epoch {epoch + 1} microbatch "
                        f"{absolute_batch + 1:,}; retrying from {batch_in_epoch:,} "
                        f"with scale={scaler.get_scale():g}",
                        flush=True,
                    )
                break

            scheduler.step()
            step += 1
            global_tokens_seen += global_tokens
            global_sequences_seen += global_sequences
            window_ce += global_ce
            window_tokens += global_tokens
            window_sequences += global_sequences

            if is_main() and log_every and step % log_every == 0:
                elapsed = time.perf_counter() - window_started
                tokens_per_second = window_tokens / elapsed if elapsed > 0 else 0.0
                print(
                    f"[train] step {step:,}/{max_steps:,} epoch={epoch + 1} "
                    f"loss={window_ce / max(1, window_tokens):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} grad={float(grad_norm):.2f} "
                    f"seq/s={window_sequences / max(elapsed, 1e-9):,.0f} "
                    f"tok/s={tokens_per_second:,.0f}",
                    flush=True,
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "train/loss": window_ce / max(1, window_tokens),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": float(grad_norm),
                            "train/tokens_per_second": tokens_per_second,
                            "train/tokens_seen": global_tokens_seen,
                        },
                        step=step,
                    )
                window_started = time.perf_counter()
                window_ce = 0.0
                window_tokens = 0
                window_sequences = 0

            if val_loader is not None and val_every and step % val_every == 0:
                improved = evaluate_primary("val")
                if improved:
                    save_best_checkpoint()
                if patience and evals_without_improve >= patience:
                    if is_main():
                        print(
                            f"[early-stop] no validation improvement for {patience} evals",
                            flush=True,
                        )
                    stop = True

            if (
                shadow_val_loader is not None
                and shadow_val_every
                and step % shadow_val_every == 0
            ):
                evaluate_shadow("shadow-val")

            if checkpoint_every and step % checkpoint_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    asdict(config),
                    str(Path(checkpoint_dir) / f"step_{step}.pt"),
                    vocab=vocab.token_to_id,
                    scheduler=scheduler,
                    scaler=scaler,
                    training_state=training_state(),
                    run_config=run_config,
                )
                if is_main():
                    print(f"[checkpoint] saved step_{step}.pt", flush=True)

            if step >= max_steps or stop:
                exhausted_epoch = False
                break

        if retry_after_overflow:
            continue
        if exhausted_epoch and batch_in_epoch >= microbatches_per_epoch:
            epoch += 1
            batch_in_epoch = 0

    if val_loader is not None and last_val_step != step:
        improved = evaluate_primary("val-final")
        if improved:
            save_best_checkpoint()
    if shadow_val_loader is not None and last_shadow_val_step != step:
        evaluate_shadow("shadow-val-final")

    save_checkpoint(
        model,
        optimizer,
        step,
        asdict(config),
        str(Path(checkpoint_dir) / "last.pt"),
        vocab=vocab.token_to_id,
        scheduler=scheduler,
        scaler=scaler,
        training_state=training_state(),
        run_config=run_config,
    )
    if is_main():
        print(
            f"[done] step={step:,}/{max_steps:,} epoch={epoch + 1} "
            f"best_val={best_val:.4f} tokens_seen={global_tokens_seen:,}",
            flush=True,
        )
    if wandb_run:
        wandb_run.finish()
    cleanup_ddp()
