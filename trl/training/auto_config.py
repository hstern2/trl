"""Corpus scanning and Chinchilla-style hyperparameter suggestion."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

from trl.data.vocab import SPECIAL_TOKENS, Vocab


@dataclass
class CorpusStats:
    n_files: int
    n_seqs: int
    n_tokens: int  # includes BOS/EOS
    avg_len: int
    p50: int
    p99: int
    max_len: int
    vocab_size: int  # includes specials


def scan_corpora(paths: list[str], min_freq: int = 1) -> tuple[CorpusStats, Vocab]:
    """One pass over the JSONL corpora: collect length stats and build a Vocab."""
    counts: Counter[str] = Counter()
    lens: list[int] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = json.loads(line)
                counts.update(tokens)
                lens.append(len(tokens) + 2)  # wrap with BOS/EOS
    lens.sort()

    token_to_id: dict[str, int] = {}
    for tok in SPECIAL_TOKENS:
        token_to_id[tok] = len(token_to_id)
    for tok, c in sorted(counts.items()):
        if c >= min_freq:
            token_to_id[tok] = len(token_to_id)

    n = len(lens)
    total = sum(lens)
    stats = CorpusStats(
        n_files=len(paths),
        n_seqs=n,
        n_tokens=total,
        avg_len=max(1, total // max(1, n)),
        p50=lens[n // 2] if n else 0,
        p99=lens[min(n - 1, int(0.99 * n))] if n else 0,
        max_len=lens[-1] if n else 0,
        vocab_size=len(token_to_id),
    )
    return stats, Vocab(token_to_id)


def _round_multiple(x: int, m: int) -> int:
    return max(m, (x + m - 1) // m * m)


def estimate_params(vocab_size: int, layers: int, d_model: int, d_ff: int) -> int:
    """Rough param count: tied embed + per-block (attn + SwiGLU FFN)."""
    return vocab_size * d_model + layers * (4 * d_model * d_model + 3 * d_model * d_ff)


def default_d_ff(d_model: int) -> int:
    """SwiGLU-native default: 8/3*d_model, rounded to multiple of 64."""
    return max(64, _round_multiple(d_model * 8 // 3, 64))


def default_lr(d_model: int) -> float:
    """Scale AdamW baseline (3e-4 at d_model=512) as 1/sqrt(d_model)."""
    return 3e-4 * math.sqrt(512 / d_model)


def suggest_config(stats: CorpusStats, gpus: int = 1) -> dict:
    """Suggest model size / lr / batch size / max_steps from corpus stats.

    Chinchilla-style target of ~20 tokens per parameter for the model shape,
    step budget at 3× that target so early stopping on val loss has room.
    """
    target_params = stats.n_tokens / 20

    candidates = [
        # (layers, d_model, heads)
        (4, 128, 4),
        (4, 192, 6),
        (4, 256, 4),
        (6, 256, 4),
        (6, 384, 6),
        (6, 512, 8),
        (8, 512, 8),
        (12, 512, 8),
        (12, 768, 12),
        (16, 768, 12),
        (24, 1024, 16),
    ]
    best = candidates[0]
    best_dist = float("inf")
    for L, D, H in candidates:
        d_ff = default_d_ff(D)
        p = estimate_params(stats.vocab_size, L, D, d_ff)
        dist = abs(math.log(p / max(1.0, target_params)))
        if dist < best_dist:
            best_dist = dist
            best = (L, D, H)

    layers, d_model, heads = best
    d_ff = default_d_ff(d_model)
    est_params = estimate_params(stats.vocab_size, layers, d_model, d_ff)

    lr = default_lr(d_model)
    per_gpu_batch = max(32, min(1024, int(100_000 / stats.avg_len / max(1, gpus))))
    per_gpu_batch = _round_multiple(per_gpu_batch, 16)
    tokens_per_step = per_gpu_batch * gpus * stats.avg_len

    target_tokens = int(20 * est_params)
    max_steps = max(1000, int(3 * target_tokens / max(1, tokens_per_step)))
    max_steps = _round_multiple(max_steps, 100)

    max_seq = int(max(64, min(512, _round_multiple(stats.p99, 16))))
    warmup_steps = min(2000, max(200, max_steps // 20))

    return {
        "layers": layers,
        "d_model": d_model,
        "heads": heads,
        "d_ff": d_ff,
        "max_seq": max_seq,
        "batch_size": per_gpu_batch,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "est_params": est_params,
        "target_tokens": target_tokens,
        "tokens_per_step": tokens_per_step,
    }
