import torch

from trl.data.vocab import BOS, EOS, PAD
from trl.generation.sampler import sample


class SpecialTokenBiasedModel:
    def eval(self) -> None:
        pass

    def __call__(self, tokens, kv_cache=None):
        batch = tokens.shape[0]
        logits = torch.full((batch, 1, 4), float("-inf"))
        logits[:, :, PAD] = 100.0
        logits[:, :, BOS] = 100.0
        logits[:, :, EOS] = 0.0
        return logits, kv_cache


def test_sample_never_emits_pad_or_bos() -> None:
    sequences = sample(
        SpecialTokenBiasedModel(), 3, max_len=4, device=torch.device("cpu")  # type: ignore[arg-type]
    )

    assert sequences == [[EOS], [EOS], [EOS]]
