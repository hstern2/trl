from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from trl.cli import app
from trl.data.dataset import IndexMetadata
from trl.data.vocab import Vocab


def test_index_command_builds_primary_and_shadow_indices(tmp_path: Path) -> None:
    vocab = Vocab({"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3, "N": 4})
    vocab_path = tmp_path / "vocab.json"
    vocab.save(str(vocab_path))
    paths: dict[str, Path] = {}
    for name, token in (("train", "C"), ("validation", "N"), ("shadow", "C")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps([token]) + "\n")
        paths[name] = path

    index_dir = tmp_path / "index"
    result = CliRunner().invoke(
        app,
        [
            "index",
            str(paths["train"]),
            "--val-data",
            str(paths["validation"]),
            "--shadow-val-data",
            str(paths["shadow"]),
            "--vocab",
            str(vocab_path),
            "--index-dir",
            str(index_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert IndexMetadata.load(index_dir / "train.index.json").rows == 1
    assert IndexMetadata.load(index_dir / "validation.index.json").rows == 1
    assert IndexMetadata.load(index_dir / "shadow_validation.index.json").rows == 1
