"""Exact held-out evaluation for self-contained pretraining checkpoints."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from trl.data.dataset import IndexedTokenDataset, get_indexed_dataloader, vocab_sha256
from trl.data.vocab import Vocab
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.training.pretrain import _evaluate, resolve_precision
from trl.training.utils import normalized_model_state, setup_ddp
from trl.training.warm_start import read_checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    indexes: Mapping[str, str | Path],
    *,
    batch_size: int = 256,
    num_workers: int = 4,
    precision: str = "auto",
) -> dict[str, Any]:
    """Evaluate named indexed corpora exactly across all initialized ranks."""
    if not indexes:
        raise ValueError("at least one validation index is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    local_rank = setup_ddp()
    distributed = dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    resolved_precision = resolve_precision(precision, device)

    source = Path(checkpoint_path).expanduser().resolve()
    checkpoint = read_checkpoint(str(source))
    checkpoint_step = int(checkpoint.get("step", 0))
    vocab = Vocab(dict(checkpoint["vocab"]))
    config = TransformerConfig(**checkpoint["config"])
    if config.vocab_size != vocab.size:
        raise ValueError(
            f"checkpoint config vocabulary size {config.vocab_size} "
            f"does not match embedded vocabulary size {vocab.size}"
        )

    state = normalized_model_state(checkpoint)
    if "embed.weight" in state and "head.weight" in state:
        if not torch.equal(state["embed.weight"], state["head.weight"]):
            raise ValueError("checkpoint embedding and head weights differ")
    model = TransformerLM(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    del checkpoint, state

    result: dict[str, Any] = {
        "checkpoint": str(source),
        "checkpoint_step": checkpoint_step,
        "precision": resolved_precision,
        "world_size": world_size,
        "batch_size_per_rank": batch_size,
        "evaluations": {},
    }
    expected_vocab_hash = vocab_sha256(vocab)
    for name, index_path in indexes.items():
        dataset = IndexedTokenDataset(Path(index_path).expanduser().resolve())
        if dataset.metadata.vocab_size != vocab.size:
            raise ValueError(
                f"{name} index vocabulary size {dataset.metadata.vocab_size} "
                f"does not match checkpoint vocabulary size {vocab.size}"
            )
        if dataset.metadata.vocab_sha256 != expected_vocab_hash:
            raise ValueError(f"{name} index vocabulary IDs do not match the checkpoint")
        if dataset.metadata.max_length > config.max_seq_len:
            raise ValueError(
                f"{name} row length {dataset.metadata.max_length} exceeds "
                f"checkpoint max_seq_len={config.max_seq_len}"
            )
        loader, sampler = get_indexed_dataloader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            num_replicas=world_size,
            rank=rank,
            seed=0,
        )
        sampler.set_epoch(0)
        if distributed:
            dist.barrier()
        started = time.monotonic()
        loss, tokens = _evaluate(model, loader, device, resolved_precision)
        if distributed:
            dist.barrier()
        elapsed = time.monotonic() - started
        result["evaluations"][name] = {
            "loss": loss,
            "tokens": tokens,
            "sequences": dataset.metadata.rows,
            "elapsed_seconds": elapsed,
            "index": str(dataset.metadata_path.resolve()),
        }
        del loader, sampler, dataset
    return result
