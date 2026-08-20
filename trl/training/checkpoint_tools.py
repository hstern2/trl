"""Utilities for combining self-contained pretraining checkpoints."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from trl.training.utils import normalized_model_state
from trl.training.warm_start import read_checkpoint


def blend_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
    weights: list[float] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a self-contained convex combination of model checkpoints.

    Optimizer, scheduler, RNG, and data-loader state are intentionally omitted:
    the result is a warm-start/evaluation artifact, not an exact-resume artifact.
    Checkpoints are loaded one at a time to keep peak host memory bounded.
    """
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    sources = [Path(path).expanduser().resolve() for path in checkpoint_paths]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"checkpoint files do not exist: {missing}")

    if weights is None or not weights:
        normalized_weights = [1.0 / len(sources)] * len(sources)
    else:
        if len(weights) != len(sources):
            raise ValueError("the number of weights must match the number of checkpoints")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("checkpoint weights must be finite and non-negative")
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError("checkpoint weights must have a positive sum")
        normalized_weights = [weight / total_weight for weight in weights]

    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    reference_config: dict[str, Any] | None = None
    reference_vocab: dict[str, int] | None = None
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    reference_dtypes: dict[str, torch.dtype] | None = None
    accumulators: dict[str, torch.Tensor] = {}
    copied_tensors: dict[str, torch.Tensor] = {}
    source_steps: list[int] = []

    for source_number, (source_path, weight) in enumerate(
        zip(sources, normalized_weights, strict=True)
    ):
        checkpoint = read_checkpoint(str(source_path))
        config = dict(checkpoint["config"])
        vocab = dict(checkpoint["vocab"])
        state = normalized_model_state(checkpoint)
        shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
        dtypes = {name: tensor.dtype for name, tensor in state.items()}

        if source_number == 0:
            reference_config = config
            reference_vocab = vocab
            reference_shapes = shapes
            reference_dtypes = dtypes
        else:
            if config != reference_config:
                raise ValueError(f"checkpoint config differs: {source_path}")
            if vocab != reference_vocab:
                raise ValueError(f"checkpoint vocabulary differs: {source_path}")
            if shapes != reference_shapes:
                raise ValueError(f"checkpoint tensor names or shapes differ: {source_path}")
            if dtypes != reference_dtypes:
                raise ValueError(f"checkpoint tensor dtypes differ: {source_path}")

        for name, tensor in state.items():
            value = tensor.detach().cpu()
            if value.is_floating_point() or value.is_complex():
                if name not in accumulators:
                    accumulator_dtype = torch.complex128 if value.is_complex() else torch.float64
                    accumulators[name] = value.to(accumulator_dtype).mul_(weight)
                else:
                    accumulators[name].add_(value, alpha=weight)
            elif name not in copied_tensors:
                copied_tensors[name] = value.clone()
            elif not torch.equal(copied_tensors[name], value):
                raise ValueError(
                    f"non-floating tensor {name!r} differs in checkpoint {source_path}"
                )
        source_steps.append(int(checkpoint.get("step", 0)))
        del checkpoint, state

    assert reference_config is not None
    assert reference_vocab is not None
    assert reference_dtypes is not None
    model_state: dict[str, torch.Tensor] = {}
    for name, dtype in reference_dtypes.items():
        if name in accumulators:
            model_state[name] = accumulators[name].to(dtype)
        else:
            model_state[name] = copied_tensors[name]

    if "embed.weight" in model_state and "head.weight" in model_state:
        if not torch.equal(model_state["embed.weight"], model_state["head.weight"]):
            raise ValueError("blended embedding and head weights are not tied-equivalent")
        model_state["head.weight"] = model_state["embed.weight"]

    output: dict[str, Any] = {
        "checkpoint_version": 1,
        "model": model_state,
        "step": max(source_steps),
        "config": reference_config,
        "vocab": reference_vocab,
        "blend": {
            "format_version": 1,
            "sources": [str(path) for path in sources],
            "source_steps": source_steps,
            "weights": normalized_weights,
            "resume_capable": False,
        },
    }

    temporary_path = Path(f"{destination}.tmp.{os.getpid()}")
    try:
        torch.save(output, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return output["blend"]
