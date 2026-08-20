from __future__ import annotations

import math
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def setup_ddp() -> int:
    """Initialize DDP when launched by torchrun and return the local rank."""
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" not in os.environ:
        return 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # Rank zero may spend tens of minutes building a first-use corpus index.
    dist.init_process_group(backend, timeout=timedelta(hours=1))
    return local_rank


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Remove torch.compile and DDP wrappers."""
    current = model
    while True:
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
        elif hasattr(current, "module"):
            current = current.module
        else:
            return current


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _all_rank_rng_states() -> list[dict[str, Any]]:
    local = capture_rng_state()
    if not dist.is_initialized():
        return [local]
    states: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(states, local)
    return [state for state in states if state is not None]


def _clean_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return unwrap_model(model).state_dict()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Any,
    path: str,
    vocab: dict[str, int] | None = None,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    training_state: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
) -> None:
    """Atomically save a checkpoint, including exact distributed resume state."""
    rng_states = _all_rank_rng_states()
    if is_main():
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "checkpoint_version": 2,
            "model": _clean_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "rng_states": rng_states,
            "world_size": len(rng_states),
        }
        if vocab is not None:
            state["vocab"] = vocab
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if training_state is not None:
            state["training_state"] = training_state
        if run_config is not None:
            state["run_config"] = run_config

        temporary_path = Path(f"{checkpoint_path}.tmp.{os.getpid()}")
        try:
            torch.save(state, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    if dist.is_initialized():
        dist.barrier()


def normalized_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return model tensors with DDP/compile wrapper prefixes removed."""
    prefixes = ("module.", "_orig_mod.")
    normalized: dict[str, torch.Tensor] = {}
    for original_name, tensor in checkpoint["model"].items():
        name = original_name
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name.removeprefix(prefix)
                    changed = True
        normalized[name] = tensor
    return normalized


# Backward-compatible private alias for callers predating the public helper.
_normalized_model_state = normalized_model_state


def load_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    """Restore a version-2 training checkpoint and this rank's RNG state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = ("model", "optimizer", "scheduler", "training_state", "rng_states")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"resume checkpoint is missing fields: {missing}")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if int(checkpoint.get("world_size", 1)) != world_size:
        raise ValueError(
            "resume checkpoint world size does not match this run: "
            f"{checkpoint.get('world_size', 1)} != {world_size}"
        )

    unwrap_model(model).load_state_dict(normalized_model_state(checkpoint), strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_states"][rank])
    return checkpoint


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load model and optional optimizer state, returning the step number."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(normalized_model_state(checkpoint), strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))
