from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from trl.data.vocab import Vocab

try:
    import orjson

    _json_loads = orjson.loads
except ImportError:  # pragma: no cover - exercised only in minimal installations
    _json_loads = json.loads

INDEX_VERSION = 1
_IO_BUFFER_ITEMS = 1_000_000


def vocab_sha256(vocab: Vocab) -> str:
    payload = json.dumps(
        vocab.token_to_id,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_records(paths: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_string in paths:
        path = Path(path_string).resolve()
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return records


@dataclass(frozen=True)
class IndexMetadata:
    version: int
    name: str
    sources: list[dict[str, Any]]
    vocab_sha256: str
    vocab_size: int
    token_dtype: str
    rows: int
    tokens: int
    min_length: int
    max_length: int
    mean_length: float
    p50_length: int
    p99_length: int
    tokens_path: str
    offsets_path: str
    lengths_path: str

    @classmethod
    def load(cls, path: str | Path) -> IndexMetadata:
        return cls(**json.loads(Path(path).read_text()))


def _percentile_from_histogram(histogram: Sequence[int], count: int, q: float) -> int:
    if count == 0:
        return 0
    target = min(count - 1, int(q * count))
    seen = 0
    for length, frequency in enumerate(histogram):
        seen += frequency
        if seen > target:
            return length
    raise AssertionError("length histogram does not contain the declared row count")


def _index_paths(index_dir: str | Path, name: str) -> dict[str, Path]:
    if not name or Path(name).name != name:
        raise ValueError("index name must be a non-empty filename component")
    root = Path(index_dir)
    return {
        "metadata": root / f"{name}.index.json",
        "tokens": root / f"{name}.tokens.bin",
        "offsets": root / f"{name}.offsets.bin",
        "lengths": root / f"{name}.lengths.bin",
    }


def _metadata_matches(
    metadata: IndexMetadata,
    paths: Sequence[str],
    vocab: Vocab,
    index_paths: dict[str, Path],
) -> bool:
    if metadata.version != INDEX_VERSION:
        return False
    if metadata.name != index_paths["metadata"].name.removesuffix(".index.json"):
        return False
    if metadata.sources != _source_records(paths):
        return False
    if metadata.vocab_sha256 != vocab_sha256(vocab):
        return False
    dtype = np.dtype(metadata.token_dtype)
    expected_sizes = {
        "tokens": metadata.tokens * dtype.itemsize,
        "offsets": metadata.rows * np.dtype("<u8").itemsize,
        "lengths": metadata.rows * np.dtype("<u2").itemsize,
    }
    return all(
        index_paths[key].is_file() and index_paths[key].stat().st_size == size
        for key, size in expected_sizes.items()
    )


def build_index(
    corpus_paths: str | Sequence[str],
    vocab: Vocab,
    index_dir: str | Path,
    name: str,
    *,
    force: bool = False,
    progress: bool = False,
) -> IndexMetadata:
    """Build or reuse a compact random-access index for JSONL token sequences.

    The binary token stream includes BOS and EOS. No row is filtered or
    truncated. Files are written under temporary names and the metadata is
    installed last, so an interrupted build is never mistaken for a complete
    index.
    """
    paths = [corpus_paths] if isinstance(corpus_paths, str) else list(corpus_paths)
    if not paths:
        raise ValueError("at least one corpus path is required")
    if sys.byteorder != "little":
        raise RuntimeError("indexed corpora currently require a little-endian host")
    if sorted(vocab.token_to_id.values()) != list(range(vocab.size)):
        raise ValueError("vocabulary IDs must be contiguous from zero")
    for expected_id, special_token in enumerate(("<pad>", "<bos>", "<eos>")):
        if vocab.token_to_id.get(special_token) != expected_id:
            raise ValueError(f"vocabulary token {special_token!r} must have ID {expected_id}")

    output = _index_paths(index_dir, name)
    output["metadata"].parent.mkdir(parents=True, exist_ok=True)
    if output["metadata"].is_file() and not force:
        metadata = IndexMetadata.load(output["metadata"])
        if _metadata_matches(metadata, paths, vocab, output):
            return metadata

    max_id = max(vocab.token_to_id.values())
    if max_id < 2**8:
        token_dtype = "<u1"
        token_typecode = "B"
    elif max_id < 2**16:
        token_dtype = "<u2"
        token_typecode = "H"
    else:
        raise ValueError("indexed corpus supports at most 65,536 vocabulary entries")

    suffix = f".tmp.{os.getpid()}"
    temporary = {key: Path(f"{path}{suffix}") for key, path in output.items()}
    row_count = 0
    token_count = 0
    histogram = [0] * (2**16)
    token_buffer: bytearray | array[int]
    token_buffer = bytearray() if token_typecode == "B" else array(token_typecode)
    offset_buffer = array("Q")
    length_buffer = array("H")
    started = time.perf_counter()
    encoded_vocab: dict[bytes, int] | None = None
    if "orjson" in globals():
        candidate = {
            orjson.dumps(token): token_id
            for token, token_id in vocab.token_to_id.items()
            if token not in ("<pad>", "<bos>", "<eos>")
        }
        if all(b"," not in encoded_token for encoded_token in candidate):
            encoded_vocab = candidate

    try:
        with (
            temporary["tokens"].open("wb", buffering=8 * 1024 * 1024) as token_handle,
            temporary["offsets"].open("wb", buffering=8 * 1024 * 1024) as offset_handle,
            temporary["lengths"].open("wb", buffering=8 * 1024 * 1024) as length_handle,
        ):
            for corpus_path in paths:
                with Path(corpus_path).open("rb", buffering=8 * 1024 * 1024) as corpus_handle:
                    for source_line, line in enumerate(corpus_handle, start=1):
                        if not line.strip():
                            raise ValueError(f"blank row in {corpus_path} at line {source_line:,}")
                        ids: list[int]
                        if encoded_vocab is not None:
                            encoded_tokens = line.rstrip(b"\r\n")[1:-1].split(b",")
                            try:
                                ids = [1, *(encoded_vocab[token] for token in encoded_tokens), 2]
                            except KeyError:
                                encoded_tokens = []
                            if encoded_tokens:
                                tokens = None
                            else:
                                tokens = _json_loads(line)
                        else:
                            tokens = _json_loads(line)
                        if tokens is not None:
                            if not isinstance(tokens, list) or not all(
                                isinstance(token, str) for token in tokens
                            ):
                                raise ValueError(
                                    f"row in {corpus_path} at line {source_line:,} "
                                    "is not a JSON list of strings"
                                )
                            try:
                                ids = [1, *(vocab.token_to_id[token] for token in tokens), 2]
                            except KeyError as error:
                                raise ValueError(
                                    f"unknown token {error.args[0]!r} in {corpus_path} "
                                    f"at line {source_line:,}"
                                ) from error
                        length = len(ids)
                        if length >= 2**16:
                            raise ValueError(
                                f"row in {corpus_path} at line {source_line:,} "
                                f"has {length:,} model tokens; maximum is 65,535"
                            )

                        offset_buffer.append(token_count)
                        length_buffer.append(length)
                        token_buffer.extend(ids)
                        row_count += 1
                        token_count += length
                        histogram[length] += 1

                        if progress and row_count % 2_000_000 == 0:
                            elapsed = time.perf_counter() - started
                            print(
                                f"[index:{name}] rows={row_count:,} "
                                f"tokens={token_count:,} "
                                f"rows/s={row_count / max(elapsed, 1e-9):,.0f}",
                                flush=True,
                            )

                        if len(token_buffer) >= _IO_BUFFER_ITEMS:
                            if isinstance(token_buffer, bytearray):
                                token_handle.write(token_buffer)
                                token_buffer = bytearray()
                            else:
                                token_buffer.tofile(token_handle)
                                token_buffer = array(token_typecode)
                        if len(offset_buffer) >= _IO_BUFFER_ITEMS:
                            offset_buffer.tofile(offset_handle)
                            offset_buffer = array("Q")
                            length_buffer.tofile(length_handle)
                            length_buffer = array("H")

            if isinstance(token_buffer, bytearray):
                token_handle.write(token_buffer)
            else:
                token_buffer.tofile(token_handle)
            offset_buffer.tofile(offset_handle)
            length_buffer.tofile(length_handle)

        if row_count == 0:
            raise ValueError("cannot build an index for an empty corpus")

        metadata = IndexMetadata(
            version=INDEX_VERSION,
            name=name,
            sources=_source_records(paths),
            vocab_sha256=vocab_sha256(vocab),
            vocab_size=vocab.size,
            token_dtype=token_dtype,
            rows=row_count,
            tokens=token_count,
            min_length=next(index for index, count in enumerate(histogram) if count),
            max_length=max(index for index, count in enumerate(histogram) if count),
            mean_length=token_count / row_count,
            p50_length=_percentile_from_histogram(histogram, row_count, 0.50),
            p99_length=_percentile_from_histogram(histogram, row_count, 0.99),
            tokens_path=output["tokens"].name,
            offsets_path=output["offsets"].name,
            lengths_path=output["lengths"].name,
        )
        temporary["metadata"].write_text(json.dumps(asdict(metadata), indent=2) + "\n")
        for key in ("tokens", "offsets", "lengths", "metadata"):
            os.replace(temporary[key], output[key])
        return metadata
    finally:
        for path in temporary.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class IndexedTokenDataset(Dataset[torch.Tensor]):
    """Memory-mapped token sequences backed by :func:`build_index`."""

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata = IndexMetadata.load(self.metadata_path)
        root = self.metadata_path.parent
        self.tokens_path = root / self.metadata.tokens_path
        self.offsets_path = root / self.metadata.offsets_path
        self.lengths_path = root / self.metadata.lengths_path
        self._tokens: np.memmap[Any, Any] | None = None
        self._offsets: np.memmap[Any, Any] | None = None
        self._lengths: np.memmap[Any, Any] | None = None
        self._validate_files()

    def _validate_files(self) -> None:
        expected = {
            self.tokens_path: self.metadata.tokens * np.dtype(self.metadata.token_dtype).itemsize,
            self.offsets_path: self.metadata.rows * np.dtype("<u8").itemsize,
            self.lengths_path: self.metadata.rows * np.dtype("<u2").itemsize,
        }
        for path, size in expected.items():
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(
                    f"indexed corpus file has the wrong size: {path} (expected {size:,} bytes)"
                )

    def _open(self) -> None:
        if self._tokens is None:
            self._tokens = np.memmap(
                self.tokens_path,
                mode="r",
                dtype=self.metadata.token_dtype,
                shape=(self.metadata.tokens,),
            )
            self._offsets = np.memmap(
                self.offsets_path,
                mode="r",
                dtype="<u8",
                shape=(self.metadata.rows,),
            )
            self._lengths = np.memmap(
                self.lengths_path,
                mode="r",
                dtype="<u2",
                shape=(self.metadata.rows,),
            )

    @property
    def lengths(self) -> np.memmap[Any, Any]:
        self._open()
        assert self._lengths is not None
        return self._lengths

    def __len__(self) -> int:
        return self.metadata.rows

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        self._open()
        assert self._tokens is not None and self._offsets is not None
        start = int(self._offsets[idx])
        length = int(self.lengths[idx])
        return torch.tensor(self._tokens[start : start + length], dtype=torch.long)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_tokens"] = None
        state["_offsets"] = None
        state["_lengths"] = None
        return state


def collate_token_sequences(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences and return next-token model inputs and targets."""
    max_len = max(sequence.numel() for sequence in batch)
    padded = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, sequence in enumerate(batch):
        padded[row, : sequence.numel()] = sequence
    return padded[:, :-1], padded[:, 1:]


class DistributedBucketBatchSampler(Sampler[list[int]]):
    """Bounded-memory deterministic shuffle, bucketing, and exact DDP sharding.

    Global batches are formed first and split across ranks, so every row is
    visited exactly once per epoch. The final local batches may differ in size
    by one row; the training loop performs token-weighted gradient reduction.
    """

    def __init__(
        self,
        dataset: IndexedTokenDataset,
        batch_size: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        bucket_width: int = 8,
        shuffle_block_size: int = 1_000_000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if bucket_width <= 0 or shuffle_block_size <= 0:
            raise ValueError("bucket and shuffle block sizes must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_width = bucket_width
        self.shuffle_block_size = shuffle_block_size
        self.epoch = 0
        self.start_batch = 0

    @property
    def global_batch_size(self) -> int:
        return self.batch_size * self.num_replicas

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = epoch
        self.start_batch = start_batch

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.global_batch_size)

    def _global_batches(self) -> Iterator[np.ndarray[Any, np.dtype[np.int64]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        starts = np.arange(0, len(self.dataset), self.shuffle_block_size, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(starts)
        bucket_count = math.ceil((self.dataset.metadata.max_length + 1) / self.bucket_width)
        pending = [np.empty(0, dtype=np.int64) for _ in range(bucket_count)]
        global_batch = self.global_batch_size

        for start_value in starts:
            start = int(start_value)
            stop = min(len(self.dataset), start + self.shuffle_block_size)
            indices = np.arange(start, stop, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)
            keys = np.asarray(self.dataset.lengths[indices], dtype=np.int64) // self.bucket_width
            order = np.argsort(keys, kind="stable")
            sorted_indices = indices[order]
            sorted_keys = keys[order]
            boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
            groups = list(np.split(sorted_indices, boundaries))
            if self.shuffle:
                rng.shuffle(groups)
            for group in groups:
                key = int(self.dataset.lengths[int(group[0])]) // self.bucket_width
                combined = np.concatenate((pending[key], group))
                complete = len(combined) // global_batch * global_batch
                for offset in range(0, complete, global_batch):
                    yield combined[offset : offset + global_batch]
                pending[key] = combined[complete:]

        tail_groups = [group for group in pending if len(group)]
        if not tail_groups:
            return
        tail = np.concatenate(tail_groups)
        if self.shuffle:
            rng.shuffle(tail)
        for offset in range(0, len(tail), global_batch):
            yield tail[offset : offset + global_batch]

    def __iter__(self) -> Iterator[list[int]]:
        for batch_number, global_indices in enumerate(self._global_batches()):
            if batch_number < self.start_batch:
                continue
            local_indices = np.array_split(global_indices, self.num_replicas)[self.rank]
            if len(local_indices) == 0:
                raise RuntimeError(
                    "the final global batch has fewer rows than DDP ranks; "
                    "use a smaller world size or a larger dataset"
                )
            yield local_indices.tolist()


def get_indexed_dataloader(
    dataset: IndexedTokenDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    num_replicas: int,
    rank: int,
    seed: int = 0,
    bucket_width: int = 8,
    shuffle_block_size: int = 1_000_000,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DistributedBucketBatchSampler]:
    sampler = DistributedBucketBatchSampler(
        dataset,
        batch_size,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        bucket_width=bucket_width,
        shuffle_block_size=shuffle_block_size,
    )
    worker_generator = torch.Generator()
    worker_generator.manual_seed(seed + rank)
    loader = cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_token_sequences,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=worker_generator,
        ),
    )
    return loader, sampler
