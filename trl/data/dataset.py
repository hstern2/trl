from __future__ import annotations

import hashlib
import json
import math
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from trl.data.vocab import EOS, PAD, Vocab


def _val_bucket(idx: int, val_fraction: float, seed: int) -> bool:
    """Deterministic hash-based split: True if this row belongs to the val set."""
    h = hashlib.blake2b(f"{seed}:{idx}".encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") % 10_000) < int(val_fraction * 10_000)


class TokenDataset(Dataset):  # type: ignore[type-arg]
    """Dataset of token sequences from one or more JSONL corpora.

    If ``val_fraction`` > 0, rows are deterministically split by hash into
    ``split='train'`` and ``split='val'`` subsets. Sequences longer than
    ``max_seq`` are dropped from whichever split they would land in.
    """

    def __init__(
        self,
        corpus_path: str | list[str],
        vocab: Vocab,
        max_seq: int = 192,
        split: str = "all",
        val_fraction: float = 0.0,
        seed: int = 0,
    ) -> None:
        assert split in ("all", "train", "val")
        self.vocab = vocab
        self.max_seq = max_seq
        self.sequences: list[list[int]] = []

        paths = [corpus_path] if isinstance(corpus_path, str) else list(corpus_path)
        row = 0
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    idx = row
                    row += 1
                    if split != "all" and val_fraction > 0:
                        is_val = _val_bucket(idx, val_fraction, seed)
                        if (split == "val") != is_val:
                            continue
                    tokens = json.loads(line)
                    ids = vocab.encode(tokens)
                    if len(ids) <= max_seq:
                        self.sequences.append(ids)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> list[int]:
        return self.sequences[idx]


def _collate(batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to max length in batch, return (input, target) tensors."""
    max_len = max(len(s) for s in batch)
    padded = [s + [PAD] * (max_len - len(s)) for s in batch]
    t = torch.tensor(padded, dtype=torch.long)
    # input: all tokens except last; target: all tokens except first
    return t[:, :-1], t[:, 1:]


class BucketedSampler(Sampler[int]):
    """Sampler that groups sequences by length into buckets for efficient batching."""

    def __init__(
        self,
        dataset: TokenDataset,
        batch_size: int,
        num_buckets: int = 8,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Sort indices by sequence length, split into buckets
        sorted_indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i]))
        bucket_size = math.ceil(len(sorted_indices) / num_buckets)
        self.buckets = [
            sorted_indices[i : i + bucket_size]
            for i in range(0, len(sorted_indices), bucket_size)
        ]

    def __iter__(self) -> Iterator[int]:
        indices: list[int] = []
        for bucket in self.buckets:
            if self.shuffle:
                perm = torch.randperm(len(bucket)).tolist()
                indices.extend(bucket[i] for i in perm)
            else:
                indices.extend(bucket)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


def get_dataloader(
    dataset: TokenDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader, optionally with DistributedSampler for DDP."""
    sampler: Sampler[int] | None = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = BucketedSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
