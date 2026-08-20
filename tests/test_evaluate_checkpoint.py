from __future__ import annotations

import json
import math

import torch

from trl.data.dataset import build_index
from trl.data.vocab import Vocab
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.training.evaluate import evaluate_checkpoint


def test_evaluate_checkpoint_on_index(tmp_path) -> None:
    vocab = Vocab({"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3, "N": 4})
    corpus = tmp_path / "validation.jsonl"
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in (["C"], ["C", "N"], ["N", "C"]))
    )
    metadata = build_index(str(corpus), vocab, tmp_path / "index", "validation")
    config = TransformerConfig(
        vocab_size=vocab.size,
        n_layers=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        max_seq_len=8,
        dropout=0.1,
    )
    model = TransformerLM(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": 7,
            "config": vars(config),
            "vocab": vocab.token_to_id,
        },
        checkpoint,
    )

    result = evaluate_checkpoint(
        checkpoint,
        {"primary": tmp_path / "index" / "validation.index.json"},
        batch_size=2,
        num_workers=0,
        precision="fp32",
    )

    evaluation = result["evaluations"]["primary"]
    assert result["checkpoint_step"] == 7
    assert evaluation["tokens"] == metadata.tokens - metadata.rows
    assert evaluation["sequences"] == metadata.rows
    assert math.isfinite(evaluation["loss"])
