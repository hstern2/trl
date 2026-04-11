from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for positions up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape (seq_len, dim//2)."""
        return (
            self.cos_cached[offset : offset + seq_len],
            self.sin_cached[offset : offset + seq_len],
        )


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (..., seq_len, dim).

    cos, sin are (seq_len, dim//2).
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    # Broadcast cos/sin to match batch dims
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
