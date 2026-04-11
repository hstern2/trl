from __future__ import annotations

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
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    loader = get_dataloader(dataset, batch_size, distributed=distributed)
    total_steps = len(loader) * epochs
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    # Optional wandb
    wandb_run = None
    if wandb_project and is_main():
        import wandb

        wandb_run = wandb.init(project=wandb_project, config=asdict(config))

    step = 0
    best_loss = float("inf")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if distributed and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            n_tokens = (targets != PAD).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            step += 1

            if wandb_run:
                wandb_run.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)

            if checkpoint_every and step % checkpoint_every == 0:
                save_checkpoint(
                    model, optimizer, step, asdict(config),
                    str(Path(checkpoint_dir) / f"step_{step}.pt"),
                )

        avg_loss = epoch_loss / max(1, epoch_tokens)
        if is_main():
            print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, step, asdict(config),
                str(Path(checkpoint_dir) / "best.pt"),
            )

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()
