from __future__ import annotations

import json
import math
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from trl.data.vocab import EOS, PAD, Vocab


class TokenDataset(Dataset):  # type: ignore[type-arg]
    """Dataset of token sequences from a JSONL corpus."""

    def __init__(self, corpus_path: str, vocab: Vocab, max_seq: int = 192) -> None:
        self.vocab = vocab
        self.max_seq = max_seq
        self.sequences: list[list[int]] = []

        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if not line:
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
        g = torch.Generator()
        if self.shuffle:
            g.manual_seed(torch.randint(0, 2**31, (1,)).item())

        # Shuffle within each bucket, then yield
        indices: list[int] = []
        for bucket in self.buckets:
            bucket = list(bucket)
            if self.shuffle:
                perm = torch.randperm(len(bucket), generator=g).tolist()
                bucket = [bucket[i] for i in perm]
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
