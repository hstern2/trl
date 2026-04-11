from __future__ import annotations

import random

from trl.objectives.base import ScoredItem


class ReplayBuffer:
    """Fixed-capacity buffer with priority sampling by reward."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.items: list[tuple[ScoredItem, float]] = []  # (item, reward)

    def add(self, items: list[ScoredItem], rewards: list[float]) -> None:
        """Add scored items with their rewards."""
        for item, reward in zip(items, rewards):
            if not item.valid:
                continue
            if len(self.items) < self.capacity:
                self.items.append((item, reward))
            else:
                # Replace lowest-reward item if new one is better
                min_idx = min(range(len(self.items)), key=lambda i: self.items[i][1])
                if reward > self.items[min_idx][1]:
                    self.items[min_idx] = (item, reward)

    def sample(self, n: int) -> list[ScoredItem]:
        """Sample n items, weighted by reward."""
        if not self.items:
            return []
        n = min(n, len(self.items))
        rewards = [max(r, 1e-6) for _, r in self.items]
        total = sum(rewards)
        weights = [r / total for r in rewards]
        chosen = random.choices(self.items, weights=weights, k=n)
        return [item for item, _ in chosen]

    def __len__(self) -> int:
        return len(self.items)
