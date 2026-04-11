from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trl.model.attention import Attention
from trl.model.positional import RotaryEmbedding


@dataclass
class TransformerConfig:
    vocab_size: int = 256
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 192
    dropout: float = 0.1


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(self.act(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model)
        self.attn = Attention(config.d_model, config.n_heads, rope, dropout=config.dropout)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, new_cache


KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class TransformerLM(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        rope = RotaryEmbedding(config.d_model // config.n_heads, config.max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, rope) for _ in range(config.n_layers)]
        )
        self.norm = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Forward pass.

        Args:
            x: (batch, seq_len) token IDs
            kv_cache: optional list of (k, v) per layer

        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kv_cache: list of (k, v) per layer
        """
        h = self.embed(x)
        new_cache: KVCache = []

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            h, c = block(h, kv_cache=layer_cache)
            new_cache.append(c)

        h = self.norm(h)
        logits = self.head(h)
        return logits, new_cache

    def hidden(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Return hidden states (before LM head) instead of logits."""
        h = self.embed(x)
        new_cache: KVCache = []

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            h, c = block(h, kv_cache=layer_cache)
            new_cache.append(c)

        return self.norm(h), new_cache
