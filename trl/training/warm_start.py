"""Checkpoint-compatible vocabulary and model warm-start helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trl.data.vocab import SPECIAL_TOKENS, Vocab
from trl.model.transformer import TransformerLM


@dataclass(frozen=True)
class WarmStartInfo:
    checkpoint_step: int
    old_vocab_size: int
    new_vocab_size: int
    added_tokens: int


def read_checkpoint(path: str) -> dict[str, Any]:
    """Read and minimally validate a self-contained training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "config", "vocab"):
        if key not in checkpoint:
            raise ValueError(f"initialization checkpoint has no {key!r}: {path}")
    return checkpoint


def merge_checkpoint_vocab(
    checkpoint: dict[str, Any], corpus_vocab: Vocab
) -> tuple[Vocab, list[str]]:
    """Preserve checkpoint token IDs and append new corpus tokens."""
    old = dict(checkpoint["vocab"])
    if sorted(old.values()) != list(range(len(old))):
        raise ValueError("checkpoint vocabulary IDs must be contiguous from zero")
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        if old.get(token) != expected_id:
            raise ValueError(f"checkpoint special token {token!r} must have ID {expected_id}")

    added = sorted(set(corpus_vocab.token_to_id) - set(old))
    merged = dict(old)
    for token in added:
        merged[token] = len(merged)
    return Vocab(merged), added


def _strip_wrapper_prefix(name: str) -> str:
    """Remove prefixes introduced by DDP and torch.compile wrappers."""
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                changed = True
    return name


def load_warm_start(
    model: TransformerLM,
    checkpoint_path: str,
    vocab: Vocab,
) -> WarmStartInfo:
    """Load model weights without loading optimizer or scheduler state.

    Depth, width, heads, and feed-forward width must match. RoPE has no learned
    position table, so ``max_seq_len`` may be increased. If the merged
    vocabulary has appended tokens, old embedding rows are copied by token ID
    and new rows retain their normal random initialization.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    old_config = checkpoint["config"]
    expected = {
        "n_layers": model.config.n_layers,
        "d_model": model.config.d_model,
        "n_heads": model.config.n_heads,
        "d_ff": model.config.d_ff,
    }
    for key, value in expected.items():
        if int(old_config[key]) != value:
            raise ValueError(
                f"checkpoint {key}={old_config[key]} is incompatible with model {key}={value}"
            )

    old_vocab = dict(checkpoint["vocab"])
    for token, old_id in old_vocab.items():
        if vocab.token_to_id.get(token) != old_id:
            raise ValueError(f"token ID changed for checkpoint token {token!r}")

    source = {_strip_wrapper_prefix(name): tensor for name, tensor in checkpoint["model"].items()}
    target = model.state_dict()
    if "embed.weight" not in source or "head.weight" not in source:
        raise ValueError("checkpoint is missing tied embedding/head weights")

    old_embedding = source["embed.weight"]
    if old_embedding.shape != source["head.weight"].shape:
        raise ValueError("checkpoint embedding and head shapes differ")
    if old_embedding.shape[0] != len(old_vocab):
        raise ValueError("checkpoint embedding rows do not match checkpoint vocabulary")
    if old_embedding.shape[1] != model.config.d_model:
        raise ValueError("checkpoint embedding width does not match model width")

    expanded_embedding = target["embed.weight"].detach().clone()
    for token, old_id in old_vocab.items():
        expanded_embedding[vocab.token_to_id[token]].copy_(old_embedding[old_id])
    source["embed.weight"] = expanded_embedding
    source["head.weight"] = expanded_embedding

    unknown = sorted(set(source) - set(target))
    missing = sorted(set(target) - set(source))
    if unknown or missing:
        raise ValueError(f"checkpoint state mismatch: unknown={unknown} missing={missing}")
    for name, tensor in source.items():
        if tensor.shape != target[name].shape:
            raise ValueError(
                f"checkpoint tensor {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(target[name].shape)}"
            )

    model.load_state_dict(source, strict=True)
    return WarmStartInfo(
        checkpoint_step=int(checkpoint.get("step", 0)),
        old_vocab_size=len(old_vocab),
        new_vocab_size=vocab.size,
        added_tokens=vocab.size - len(old_vocab),
    )


def checkpoint_sha256(path: str) -> str:
    """Return a checkpoint digest for run provenance."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
