from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from trl.data.dataset import TokenDataset, _collate, get_dataloader
from trl.data.vocab import PAD, Vocab
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.training.utils import (
    cleanup_ddp,
    get_lr_scheduler,
    is_main,
    save_checkpoint,
    setup_ddp,
)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Return mean CE loss (nats/token) over the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, _ = model(inputs)
        loss = loss_fn(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        n = (targets != PAD).sum().item()
        total_loss += loss.item()
        total_tokens += n
    model.train()

    if torch.distributed.is_initialized():
        t = torch.tensor([total_loss, float(total_tokens)], device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_loss, total_tokens = t[0].item(), int(t[1].item())

    return total_loss / max(1, total_tokens)


def pretrain(
    data_path: str,
    vocab_path: str,
    layers: int = 8,
    d_model: int = 512,
    heads: int = 8,
    d_ff: int | None = None,
    max_seq: int = 192,
    dropout: float = 0.1,
    max_steps: int = 50_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    grad_clip: float = 1.0,
    z_loss: float = 1e-4,
    val_fraction: float = 0.01,
    val_every: int = 500,
    patience: int = 10,
    compile_model: bool = True,
    checkpoint_dir: str = "checkpoints/",
    checkpoint_every: int = 5000,
    log_every: int = 50,
    wandb_project: str | None = None,
) -> None:
    local_rank = setup_ddp()
    distributed = torch.distributed.is_initialized()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.load(vocab_path)
    train_ds = TokenDataset(
        data_path, vocab, max_seq=max_seq, split="train", val_fraction=val_fraction
    )
    val_ds: TokenDataset | None = None
    if val_fraction > 0:
        val_ds = TokenDataset(
            data_path, vocab, max_seq=max_seq, split="val", val_fraction=val_fraction
        )

    # SwiGLU-native default: 8/3 d_model (matches param count of a classic
    # 4×d_model FFN with gated projections, rounded to a multiple of 64).
    if d_ff is None:
        d_ff = max(64, (d_model * 8 // 3 + 63) // 64 * 64)

    config = TransformerConfig(
        vocab_size=vocab.size,
        n_layers=layers,
        d_model=d_model,
        n_heads=heads,
        d_ff=d_ff,
        max_seq_len=max_seq,
        dropout=dropout,
    )

    model = TransformerLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if distributed:
        model = DDP(model, device_ids=[local_rank])
    if compile_model and device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    train_loader = get_dataloader(train_ds, batch_size, distributed=distributed)
    val_loader: DataLoader | None = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=0,
            pin_memory=True,
        )

    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    use_amp = device.type == "cuda"

    steps_per_pass = len(train_loader)
    if is_main():
        ws = torch.distributed.get_world_size() if distributed else 1
        print(
            f"[setup] params={n_params / 1e6:.2f}M vocab={vocab.size} "
            f"layers={layers} d_model={d_model} d_ff={d_ff} heads={heads} max_seq={max_seq}",
            flush=True,
        )
        val_sz = len(val_ds) if val_ds is not None else 0
        est_passes = max_steps / max(1, steps_per_pass)
        print(
            f"[setup] train={len(train_ds):,} val={val_sz:,} seqs  batch={batch_size}×{ws}  "
            f"steps/pass={steps_per_pass}  max_steps={max_steps}  "
            f"(≈{est_passes:.2f} passes over train)  "
            f"warmup={warmup_steps}  lr={lr:g}  z_loss={z_loss:g}  "
            f"compile={compile_model}  device={device}",
            flush=True,
        )

    wandb_run = None
    if wandb_project and is_main():
        import wandb

        wandb_run = wandb.init(project=wandb_project, config=asdict(config))

    step = 0
    best_val = float("inf")
    evals_without_improve = 0
    stop = False
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    t_window = time.perf_counter()
    tokens_window = 0
    loss_window = 0.0
    loss_window_tokens = 0

    data_pass = 0
    while step < max_steps and not stop:
        data_pass += 1
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(data_pass - 1)

        if is_main():
            print(f"[pass {data_pass}] start at step {step}/{max_steps}", flush=True)

        model.train()
        pass_loss = 0.0
        pass_tokens = 0

        for inputs, targets in train_loader:
            if step >= max_steps:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(inputs)
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_targets = targets.reshape(-1)
                ce = loss_fn(flat_logits, flat_targets)
                if z_loss > 0:
                    valid = flat_targets != PAD
                    lse = torch.logsumexp(flat_logits.float()[valid], dim=-1)
                    zl = z_loss * (lse * lse).mean()
                    loss = ce + zl
                else:
                    loss = ce

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            n_tokens = (targets != PAD).sum().item()
            ce_item = ce.item()
            pass_loss += ce_item * n_tokens
            pass_tokens += n_tokens
            loss_window += ce_item * n_tokens
            loss_window_tokens += n_tokens
            tokens_window += n_tokens
            step += 1

            if wandb_run:
                wandb_run.log(
                    {"loss": ce_item, "lr": scheduler.get_last_lr()[0]}, step=step
                )

            if is_main() and log_every and step % log_every == 0:
                dt = time.perf_counter() - t_window
                tps = tokens_window / dt if dt > 0 else 0.0
                avg_loss_window = loss_window / max(1, loss_window_tokens)
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"[pass {data_pass}] step {step}/{max_steps}  "
                    f"loss={avg_loss_window:.4f}  lr={cur_lr:.2e}  "
                    f"grad={grad_norm:.2f}  tok/s={tps:,.0f}",
                    flush=True,
                )
                t_window = time.perf_counter()
                tokens_window = 0
                loss_window = 0.0
                loss_window_tokens = 0

            if val_loader is not None and val_every and step % val_every == 0:
                val_loss = _evaluate(model, val_loader, device, use_amp)
                if is_main():
                    marker = "  *new best*" if val_loss < best_val else ""
                    print(
                        f"[val] step {step}  val_loss={val_loss:.4f}  "
                        f"best={min(val_loss, best_val):.4f}{marker}",
                        flush=True,
                    )
                    if wandb_run:
                        wandb_run.log({"val_loss": val_loss}, step=step)
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    evals_without_improve = 0
                    save_checkpoint(
                        model, optimizer, step, asdict(config),
                        str(Path(checkpoint_dir) / "best.pt"),
                        vocab=vocab.token_to_id,
                    )
                    if is_main():
                        print(f"[ckpt] saved best.pt (val={best_val:.4f})", flush=True)
                else:
                    evals_without_improve += 1
                    if patience and evals_without_improve >= patience:
                        if is_main():
                            print(
                                f"[early-stop] val loss has not improved for "
                                f"{patience} evals (best={best_val:.4f})",
                                flush=True,
                            )
                        stop = True
                        break

            if checkpoint_every and step % checkpoint_every == 0:
                save_checkpoint(
                    model, optimizer, step, asdict(config),
                    str(Path(checkpoint_dir) / f"step_{step}.pt"),
                    vocab=vocab.token_to_id,
                )
                if is_main():
                    print(f"[ckpt] saved step_{step}.pt", flush=True)

        if pass_tokens > 0:
            avg_loss = pass_loss / pass_tokens
            if is_main():
                print(
                    f"[pass {data_pass}] done at step {step}/{max_steps}  "
                    f"avg_train_loss={avg_loss:.4f}  tokens={pass_tokens:,}",
                    flush=True,
                )
            # Fallback: if no validation split, track best by train-pass loss.
            if val_loader is None and avg_loss < best_val:
                best_val = avg_loss
                save_checkpoint(
                    model, optimizer, step, asdict(config),
                    str(Path(checkpoint_dir) / "best.pt"),
                    vocab=vocab.token_to_id,
                )
                if is_main():
                    print(f"[ckpt] saved best.pt (train={best_val:.4f})", flush=True)

    if is_main():
        print(
            f"[done] step {step}/{max_steps}  best_val={best_val:.4f}  "
            f"passes={data_pass}",
            flush=True,
        )

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()
