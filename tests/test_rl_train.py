import numpy as np
import pytest
import torch

from trl.data.vocab import BOS, EOS, PAD
from trl.objectives.base import Objective, Objectives, ScoredItem
from trl.training.rl_train import (
    _global_pareto_rewards,
    _masked_mean,
    _prepare_sequences,
    _score_summary,
)


class FirstObjective(Objective):
    def __init__(self) -> None:
        super().__init__("first", direction="maximize")

    def score_batch(self, items):
        return [float(item) for item in items]


class SecondObjective(Objective):
    def __init__(self) -> None:
        super().__init__("second", direction="maximize")

    def score_batch(self, items):
        return [float(item) for item in items]


def _objectives() -> Objectives:
    return Objectives([FirstObjective(), SecondObjective()], decode_fn=lambda tokens: tokens)


def test_prepare_sequences_prepends_bos_and_masks_pad_not_eos() -> None:
    token_ids, mask = _prepare_sequences([[7, EOS], [8]], torch.device("cpu"))

    assert token_ids.tolist() == [[BOS, 7, EOS], [BOS, 8, PAD]]
    assert mask.tolist() == [[True, True], [True, False]]


def test_masked_mean_excludes_padded_positions() -> None:
    values = torch.tensor([[1.0, 99.0], [3.0, 99.0]])
    mask = torch.tensor([[True, False], [True, False]])
    assert _masked_mean(values, mask).item() == pytest.approx(2.0)


def test_global_pareto_rewards_leave_invalid_items_at_zero() -> None:
    scored = [
        ScoredItem(token_ids=[], scores={"first": 2.0, "second": 1.0}),
        ScoredItem(token_ids=[], scores={"first": 1.0, "second": 2.0}),
        ScoredItem(token_ids=[], valid=False, rejection_reason="rejected"),
    ]

    rewards, gathered = _global_pareto_rewards(_objectives(), scored)

    assert len(gathered) == 3
    assert rewards[:2].min() > 0
    assert np.array_equal(rewards[2:], np.zeros(1))


def test_score_summary_reports_objectives_and_rejection_reasons() -> None:
    scored = [
        ScoredItem(token_ids=[], scores={"first": 2.0, "second": 4.0}),
        ScoredItem(token_ids=[], valid=False, rejection_reason="PoseBusters failed"),
        ScoredItem(token_ids=[], valid=False, rejection_reason="PoseBusters failed"),
    ]

    summary = _score_summary(scored, _objectives())

    assert summary["validity"] == pytest.approx(1 / 3)
    assert float(summary["reward_mean"]) > 0
    assert summary["objective_means"] == {"first": 2.0, "second": 4.0}
    assert summary["rejections"] == {"PoseBusters failed": 2}
