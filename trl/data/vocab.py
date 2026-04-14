from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PAD = 0
BOS = 1
EOS = 2
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


class Vocab:
    """Token string <-> integer mapping with special tokens."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.size = len(token_to_id)

    @classmethod
    def build(cls, corpus_path: str | list[str], min_freq: int = 1) -> Vocab:
        """Build vocabulary from one or more JSONL corpora.

        Each line is a JSON list of token strings.
        """
        paths = [corpus_path] if isinstance(corpus_path, str) else list(corpus_path)
        counts: Counter[str] = Counter()
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = json.loads(line)
                    counts.update(tokens)

        token_to_id: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)

        for tok, count in sorted(counts.items()):
            if count >= min_freq and tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

        return cls(token_to_id)

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert token strings to integer IDs, wrapped with BOS/EOS."""
        return [BOS] + [self.token_to_id[t] for t in tokens] + [EOS]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert integer IDs back to token strings, stripping special tokens."""
        return [
            self.id_to_token[i]
            for i in ids
            if i not in (PAD, BOS, EOS)
        ]

    def save(self, path: str) -> None:
        """Save vocabulary to JSON."""
        Path(path).write_text(json.dumps(self.token_to_id, indent=2))

    @classmethod
    def load(cls, path: str) -> Vocab:
        """Load vocabulary from JSON."""
        token_to_id = json.loads(Path(path).read_text())
        return cls(token_to_id)
