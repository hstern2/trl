from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

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


def test_start_and_shadow_validation_preserve_warm_start_baseline(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    train_index, vocab = _training_fixture(tmp_path)
    corpus = tmp_path / "train.jsonl"
    build_index(corpus.as_posix(), vocab, tmp_path / "index", "validation")
    build_index(corpus.as_posix(), vocab, tmp_path / "index", "shadow_validation")

    losses = iter((1.0, 1.1, 1.2, 1.3))

    def fake_evaluate(*_args: Any, **_kwargs: Any) -> tuple[float, int]:
        return next(losses), 42

    pretrain_module = importlib.import_module("trl.training.pretrain")
    monkeypatch.setattr(pretrain_module, "_evaluate", fake_evaluate)
    checkpoint_dir = tmp_path / "validated"
    pretrain(
        train_index=str(train_index),
        val_index=str(tmp_path / "index" / "validation.index.json"),
        shadow_val_index=str(tmp_path / "index" / "shadow_validation.index.json"),
        vocab=vocab,
        layers=1,
        d_model=16,
        heads=4,
        d_ff=32,
        max_seq=16,
        dropout=0.1,
        max_steps=1,
        batch_size=2,
        grad_accum_steps=2,
        lr=1e-3,
        warmup_steps=1,
        val_every=0,
        shadow_val_every=0,
        val_at_start=True,
        precision="fp32",
        compile_model=False,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=0,
        log_every=0,
        num_workers=0,
        seed=123,
    )

    output = capsys.readouterr().out
    assert "[val-start] step 0 loss=1.0000" in output
    assert "[shadow-val-start] step 0 loss=1.1000" in output
    assert "[val-final] step 1 loss=1.2000" in output
    assert "[shadow-val-final] step 1 loss=1.3000" in output

    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=False)
    assert best["step"] == 0
    assert last["training_state"]["best_val"] == 1.0
    assert last["training_state"]["last_val_step"] == 1
    assert last["training_state"]["last_shadow_val_step"] == 1
    assert "shadow_val_index_sha256" in last["run_config"]
    assert last["run_config"]["shadow_val_every"] == 0
    assert last["run_config"]["val_at_start"] is True
