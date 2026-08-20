from __future__ import annotations

import torch

from trl.model.transformer import TransformerConfig, TransformerLM
from trl.training.checkpoint_tools import blend_checkpoints


def _write_checkpoint(path, value: float, step: int, *, wrapped: bool = False) -> None:
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3}
    config = TransformerConfig(
        vocab_size=len(vocab),
        n_layers=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        max_seq_len=8,
        dropout=0.1,
    )
    model = TransformerLM(config)
    state = model.state_dict()
    for tensor in state.values():
        tensor.fill_(value)
    if wrapped:
        state = {f"module._orig_mod.{name}": tensor for name, tensor in state.items()}
    torch.save(
        {
            "model": state,
            "step": step,
            "config": vars(config),
            "vocab": vocab,
        },
        path,
    )


def test_blend_checkpoints_writes_weighted_warm_start(tmp_path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "blend.pt"
    _write_checkpoint(first, 1.0, 10)
    _write_checkpoint(second, 3.0, 20, wrapped=True)

    provenance = blend_checkpoints([first, second], output, [0.25, 0.75])
    blended = torch.load(output, map_location="cpu", weights_only=False)

    assert provenance["weights"] == [0.25, 0.75]
    assert provenance["source_steps"] == [10, 20]
    assert blended["step"] == 20
    assert "optimizer" not in blended
    assert blended["blend"]["resume_capable"] is False
    for tensor in blended["model"].values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 2.5))
    assert (
        blended["model"]["embed.weight"].untyped_storage().data_ptr()
        == blended["model"]["head.weight"].untyped_storage().data_ptr()
    )


def test_blend_checkpoints_refuses_overwrite(tmp_path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "blend.pt"
    _write_checkpoint(source, 1.0, 10)
    blend_checkpoints([source], output)

    try:
        blend_checkpoints([source], output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output should not be replaced without overwrite=True")
