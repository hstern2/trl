from __future__ import annotations

import json
from pathlib import Path

import torch

from trl.data.dataset import build_index
from trl.data.vocab import Vocab
from trl.training.pretrain import pretrain, resolve_precision


def _training_fixture(tmp_path: Path) -> tuple[Path, Vocab]:
    vocab = Vocab({"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3, "N": 4})
    corpus = tmp_path / "train.jsonl"
    rows = [["C", "N"] * (index % 3 + 1) for index in range(12)]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    build_index(str(corpus), vocab, tmp_path / "index", "train")
    return tmp_path / "index" / "train.index.json", vocab


def _run(
    index: Path,
    vocab: Vocab,
    checkpoint_dir: Path,
    resume: Path | None = None,
) -> None:
    pretrain(
        train_index=str(index),
        val_index=None,
        vocab=vocab,
        layers=1,
        d_model=16,
        heads=4,
        d_ff=32,
        max_seq=16,
        dropout=0.1,
        max_steps=2,
        batch_size=2,
        grad_accum_steps=2,
        lr=1e-3,
        warmup_steps=1,
        precision="fp32",
        compile_model=False,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=1,
        log_every=0,
        num_workers=0,
        seed=123,
        resume_checkpoint=str(resume) if resume else None,
    )


def test_exact_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    index, vocab = _training_fixture(tmp_path)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    _run(index, vocab, uninterrupted)
    _run(index, vocab, resumed, uninterrupted / "step_1.pt")

    expected = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
    actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
    assert actual["step"] == expected["step"] == 2
    assert actual["training_state"] == expected["training_state"]
    for name, tensor in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], tensor, rtol=0, atol=0)


def test_cpu_auto_precision_is_fp32() -> None:
    assert resolve_precision("auto", torch.device("cpu")) == "fp32"
