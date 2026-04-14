from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from trl.data.dataset import TokenDataset, get_dataloader
from trl.data.vocab import PAD, Vocab
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.training.utils import (
    cleanup_ddp,
    get_lr_scheduler,
    is_main,
    save_checkpoint,
    setup_ddp,
)


def pretrain(
    data_path: str,
    vocab_path: str,
    layers: int = 8,
    d_model: int = 512,
    heads: int = 8,
    max_seq: int = 192,
    dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    grad_clip: float = 1.0,
    checkpoint_dir: str = "checkpoints/",
    checkpoint_every: int = 5000,
    log_every: int = 50,
    wandb_project: str | None = None,
) -> None:
    local_rank = setup_ddp()
    distributed = torch.distributed.is_initialized()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.load(vocab_path)
    dataset = TokenDataset(data_path, vocab, max_seq=max_seq)

    config = TransformerConfig(
        vocab_size=vocab.size,
        n_layers=layers,
        d_model=d_model,
        n_heads=heads,
        d_ff=d_model * 4,
        max_seq_len=max_seq,
        dropout=dropout,
    )

    model = TransformerLM(config).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    loader = get_dataloader(dataset, batch_size, distributed=distributed)
    total_steps = len(loader) * epochs
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    use_amp = device.type == "cuda"

    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(loader)
    if is_main():
        ws = torch.distributed.get_world_size() if distributed else 1
        print(
            f"[setup] params={n_params / 1e6:.2f}M vocab={vocab.size} "
            f"layers={layers} d_model={d_model} heads={heads} max_seq={max_seq}",
            flush=True,
        )
        print(
            f"[setup] dataset={len(dataset)} seqs  batch={batch_size}×{ws}  "
            f"steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
            f"warmup={warmup_steps}  lr={lr:g}  device={device}",
            flush=True,
        )

    # Optional wandb
    wandb_run = None
    if wandb_project and is_main():
        import wandb

        wandb_run = wandb.init(project=wandb_project, config=asdict(config))

    step = 0
    best_loss = float("inf")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    t_window = time.perf_counter()
    tokens_window = 0
    loss_window = 0.0
    loss_window_tokens = 0

    for epoch in range(epochs):
        if distributed and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        if is_main():
            print(f"[epoch {epoch + 1}/{epochs}] start", flush=True)

        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            n_tokens = (targets != PAD).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            loss_window += loss.item() * n_tokens
            loss_window_tokens += n_tokens
            tokens_window += n_tokens
            step += 1

            if wandb_run:
                wandb_run.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)

            if is_main() and log_every and step % log_every == 0:
                dt = time.perf_counter() - t_window
                tps = tokens_window / dt if dt > 0 else 0.0
                avg_loss_window = loss_window / max(1, loss_window_tokens)
                cur_lr = scheduler.get_last_lr()[0]
                epoch_step = (step - 1) % steps_per_epoch + 1
                print(
                    f"[epoch {epoch + 1}/{epochs}] step {step}/{total_steps} "
                    f"({epoch_step}/{steps_per_epoch})  loss={avg_loss_window:.4f}  "
                    f"lr={cur_lr:.2e}  grad={grad_norm:.2f}  tok/s={tps:,.0f}",
                    flush=True,
                )
                t_window = time.perf_counter()
                tokens_window = 0
                loss_window = 0.0
                loss_window_tokens = 0

            if checkpoint_every and step % checkpoint_every == 0:
                save_checkpoint(
                    model, optimizer, step, asdict(config),
                    str(Path(checkpoint_dir) / f"step_{step}.pt"),
                    vocab=vocab.token_to_id,
                )
                if is_main():
                    print(f"[ckpt] saved step_{step}.pt", flush=True)

        avg_loss = epoch_loss / max(1, epoch_tokens)
        if is_main():
            print(
                f"[epoch {epoch + 1}/{epochs}] done  avg_loss={avg_loss:.4f}  "
                f"tokens={epoch_tokens:,}",
                flush=True,
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, step, asdict(config),
                str(Path(checkpoint_dir) / "best.pt"),
                vocab=vocab.token_to_id,
            )
            if is_main():
                print(f"[ckpt] saved best.pt (loss={best_loss:.4f})", flush=True)

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()
