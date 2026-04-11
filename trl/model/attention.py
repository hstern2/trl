from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trl.model.positional import RotaryEmbedding, apply_rotary

try:
    from flash_attn import flash_attn_func

    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional Flash Attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            kv_cache: optional (k, v) each (batch, n_heads, cached_len, head_dim)

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: (k, v) tensors
        """
        B, T, _ = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # q, k, v: (B, T, n_heads, head_dim)

        # Apply RoPE
        offset = 0 if kv_cache is None else kv_cache[0].shape[2]
        cos, sin = self.rope(T, offset=offset)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Concatenate with cache
        if kv_cache is not None:
            # Cache is (B, n_heads, cached_len, head_dim) -> transpose to (B, cached_len, n_heads, head_dim)
            prev_k = kv_cache[0].transpose(1, 2)
            prev_v = kv_cache[1].transpose(1, 2)
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        # Store new cache as (B, n_heads, total_len, head_dim)
        new_cache = (k.transpose(1, 2), v.transpose(1, 2))

        if HAS_FLASH and x.is_cuda and self.training:
            # flash_attn_func expects (B, S, H, D) and handles causal masking
            out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
        else:
            # Transpose to (B, n_heads, seq, head_dim) for scaled_dot_product_attention
            q = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k_t, v_t,
                is_causal=(kv_cache is None),
                dropout_p=self.dropout if self.training else 0.0,
            )
            out = out.transpose(1, 2)  # Back to (B, T, n_heads, head_dim)

        out = out.reshape(B, T, -1)
        return self.out(out), new_cache
