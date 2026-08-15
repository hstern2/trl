from __future__ import annotations

import torch

from trl.data.vocab import Vocab
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.training.warm_start import load_warm_start, merge_checkpoint_vocab


def _checkpoint(path: str) -> tuple[TransformerLM, dict[str, int]]:
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3, "N": 4}
    config = TransformerConfig(
        vocab_size=len(vocab),
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=32,
        dropout=0.1,
    )
    model = TransformerLM(config)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {},
            "step": 123,
            "config": {
                "vocab_size": len(vocab),
                "n_layers": 2,
                "d_model": 32,
                "n_heads": 4,
                "d_ff": 64,
                "max_seq_len": 32,
                "dropout": 0.1,
            },
            "vocab": vocab,
        },
        path,
    )
    return model, vocab


def test_warm_start_preserves_ids_and_appends_embedding(tmp_path) -> None:
    path = str(tmp_path / "init.pt")
    old_model, old_vocab = _checkpoint(path)
    corpus_vocab = Vocab({"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3, "O": 4})
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vocab, added = merge_checkpoint_vocab(checkpoint, corpus_vocab)

    assert vocab.token_to_id["C"] == old_vocab["C"]
    assert vocab.token_to_id["N"] == old_vocab["N"]
    assert vocab.token_to_id["O"] == len(old_vocab)
    assert added == ["O"]

    config = TransformerConfig(
        vocab_size=vocab.size,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=64,
        dropout=0.1,
    )
    new_model = TransformerLM(config)
    new_row_before = new_model.embed.weight[vocab.token_to_id["O"]].detach().clone()
    info = load_warm_start(new_model, path, vocab)

    assert info.checkpoint_step == 123
    assert info.added_tokens == 1
    for token, old_id in old_vocab.items():
        new_id = vocab.token_to_id[token]
        torch.testing.assert_close(new_model.embed.weight[new_id], old_model.embed.weight[old_id])
    torch.testing.assert_close(new_model.embed.weight[vocab.token_to_id["O"]], new_row_before)
    torch.testing.assert_close(
        new_model.blocks[0].attn.qkv.weight, old_model.blocks[0].attn.qkv.weight
    )
    assert new_model.head.weight.data_ptr() == new_model.embed.weight.data_ptr()
