from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from trl.objectives.pareto import pareto_rank_rewards


@dataclass
class ScoredItem:
    token_ids: list[int]
    log_prob: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    rejection_reason: str = ""


class Objective(ABC):
    """Abstract base class for a scoring objective."""

    def __init__(
        self,
        name: str,
        direction: str = "maximize",
        reject_above: float | None = None,
        reject_below: float | None = None,
    ) -> None:
        self.name = name
        self.direction = direction
        self.reject_above = reject_above
        self.reject_below = reject_below

    @abstractmethod
    def score_batch(self, items: list[Any]) -> list[float]:
        """Score a batch of decoded items. Return one float per item."""
        ...

    def reject(self, score: float) -> tuple[bool, str]:
        """Check if a score should be rejected."""
        if self.reject_above is not None and score > self.reject_above:
            return True, f"{self.name}={score:.3f} > {self.reject_above}"
        if self.reject_below is not None and score < self.reject_below:
            return True, f"{self.name}={score:.3f} < {self.reject_below}"
        return False, ""

    def normalize(self, scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1]."""
        arr = np.array(scores)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return [0.5] * len(scores)
        normed = ((arr - lo) / (hi - lo)).tolist()
        if self.direction == "minimize":
            normed = [1.0 - x for x in normed]
        return normed


class Objectives:
    """Container for multiple objectives with a decode bridge."""

    def __init__(
        self,
        objectives: list[Objective],
        decode_fn: Callable[[list[str]], Any | None],
        pareto_lambda: float = 0.1,
        extra_rejection_fn: Callable[[Any], tuple[bool, str]] | None = None,
    ) -> None:
        self.objectives = objectives
        self.decode_fn = decode_fn
        self.pareto_lambda = pareto_lambda
        self.extra_rejection_fn = extra_rejection_fn

    def evaluate(self, token_sequences: list[list[str]]) -> list[ScoredItem]:
        """Decode and score a batch of token sequences."""
        # Decode all sequences
        decoded = [self.decode_fn(seq) for seq in token_sequences]

        # Build ScoredItems
        items: list[ScoredItem] = []
        for i, obj in enumerate(decoded):
            item = ScoredItem(token_ids=[])  # token_ids set by caller
            if obj is None:
                item.valid = False
                item.rejection_reason = "decode failed"
            elif self.extra_rejection_fn is not None:
                rejected, reason = self.extra_rejection_fn(obj)
                if rejected:
                    item.valid = False
                    item.rejection_reason = reason
            items.append(item)

        # Score valid items with each objective
        valid_indices = [i for i, item in enumerate(items) if item.valid]
        valid_objects = [decoded[i] for i in valid_indices]

        for objective in self.objectives:
            if not valid_objects:
                break
            scores = objective.score_batch(valid_objects)
            for idx, score in zip(valid_indices, scores):
                items[idx].scores[objective.name] = score
                rejected, reason = objective.reject(score)
                if rejected:
                    items[idx].valid = False
                    items[idx].rejection_reason = reason

        return items

    def get_rewards(self, scored: list[ScoredItem]) -> np.ndarray:
        """Compute scalar rewards from scored items using Pareto ranking."""
        obj_names = [o.name for o in self.objectives]
        directions = [o.direction for o in self.objectives]

        # Build score matrix: (n_items, n_objectives), normalized
        n = len(scored)
        m = len(self.objectives)
        raw_scores = np.zeros((n, m))
        valid_mask = np.array([s.valid for s in scored])

        for j, obj in enumerate(self.objectives):
            col = [scored[i].scores.get(obj.name, 0.0) for i in range(n)]
            normed = obj.normalize(col)
            raw_scores[:, j] = normed

        # Invalid items get zero reward
        rewards = np.zeros(n)
        if valid_mask.any():
            valid_scores = raw_scores[valid_mask]
            valid_rewards = pareto_rank_rewards(valid_scores, self.pareto_lambda)
            rewards[valid_mask] = valid_rewards

        return rewards
