from __future__ import annotations

import torch
import torch.nn.functional as F

from trl.data.vocab import BOS, EOS
from trl.model.transformer import KVCache, TransformerLM


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits with top-k and/or nucleus (top-p) sampling."""
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def sample(
    model: TransformerLM,
    n: int,
    max_len: int = 192,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device | None = None,
) -> list[list[int]]:
    """Sample n sequences from the model using KV-cache decoding.

    Returns list of token ID sequences (excluding BOS, including EOS).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    tokens = torch.full((n, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    sequences: list[list[int]] = [[] for _ in range(n)]
    cache: KVCache | None = None

    for _ in range(max_len - 1):
        logits, cache = model(tokens, kv_cache=cache)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        for i in range(n):
            if not finished[i]:
                tok = next_token[i].item()
                sequences[i].append(tok)
                if tok == EOS:
                    finished[i] = True

        if finished.all():
            break

        tokens = next_token

    return sequences
