from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from trl.data.dataset import (
    DistributedBucketBatchSampler,
    IndexedTokenDataset,
    build_index,
    get_indexed_dataloader,
)
from trl.data.vocab import Vocab


def _write_corpus(path: Path, rows: list[list[str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _vocab() -> Vocab:
    return Vocab({"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3, "N": 4, "O": 5})


def test_index_round_trip_and_loader(tmp_path: Path) -> None:
    rows = [["C"], ["C", "N"], ["O", "C", "N"], ["N"]]
    corpus = tmp_path / "tokens.jsonl"
    _write_corpus(corpus, rows)

    metadata = build_index(str(corpus), _vocab(), tmp_path / "index", "train")
    assert metadata.rows == len(rows)
    assert metadata.tokens == sum(len(row) + 2 for row in rows)
    assert metadata.min_length == 3
    assert metadata.max_length == 5

    dataset = IndexedTokenDataset(tmp_path / "index" / "train.index.json")
    assert dataset[0].tolist() == [1, 3, 2]
    assert dataset[-1].tolist() == [1, 4, 2]

    loader, _ = get_indexed_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        num_replicas=1,
        rank=0,
    )
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_index_rejects_unknown_tokens(tmp_path: Path) -> None:
    corpus = tmp_path / "tokens.jsonl"
    _write_corpus(corpus, [["C"], ["X"]])
    with pytest.raises(ValueError, match="unknown token 'X'"):
        build_index(str(corpus), _vocab(), tmp_path / "index", "train")


def test_index_rejects_invalid_special_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "tokens.jsonl"
    _write_corpus(corpus, [["C"]])
    invalid = Vocab({"<bos>": 0, "<pad>": 1, "<eos>": 2, "C": 3})
    with pytest.raises(ValueError, match="'<pad>' must have ID 0"):
        build_index(str(corpus), invalid, tmp_path / "index", "train")


def test_distributed_sampler_covers_every_row_once(tmp_path: Path) -> None:
    rows = [["C"] * (index % 7 + 1) for index in range(103)]
    corpus = tmp_path / "tokens.jsonl"
    _write_corpus(corpus, rows)
    build_index(str(corpus), _vocab(), tmp_path / "index", "train")
    dataset = IndexedTokenDataset(tmp_path / "index" / "train.index.json")

    samplers = [
        DistributedBucketBatchSampler(
            dataset,
            batch_size=8,
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=17,
            shuffle_block_size=23,
        )
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]
    assert len(rank_batches[0]) == len(rank_batches[1]) == len(samplers[0])

    visited: list[int] = []
    for batches in zip(*rank_batches, strict=True):
        visited.extend(batches[0])
        visited.extend(batches[1])
    assert sorted(visited) == list(range(len(rows)))

    full = rank_batches[0]
    samplers[0].set_epoch(0, start_batch=2)
    assert list(samplers[0]) == full[2:]
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != full
