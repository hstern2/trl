from __future__ import annotations

import numpy as np


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a Pareto-dominates b (all >= and at least one >)."""
    return bool(np.all(a >= b) and np.any(a > b))


def nsga2_sort(scores: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """NSGA-II non-dominated sorting with crowding distance.

    Args:
        scores: (n, m) array of objective values (all maximized).

    Returns:
        fronts: list of lists of indices, front 0 is Pareto-optimal
        crowding: (n,) array of crowding distances
    """
    n = scores.shape[0]
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(scores[i], scores[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif _dominates(scores[j], scores[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    # Build subsequent fronts
    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    # Remove trailing empty front
    if not fronts[-1]:
        fronts.pop()

    # Crowding distance
    crowding = np.zeros(n)
    m = scores.shape[1]

    for front in fronts:
        if len(front) <= 2:
            for i in front:
                crowding[i] = float("inf")
            continue

        for obj_idx in range(m):
            sorted_front = sorted(front, key=lambda i: scores[i, obj_idx])
            crowding[sorted_front[0]] = float("inf")
            crowding[sorted_front[-1]] = float("inf")

            obj_range = scores[sorted_front[-1], obj_idx] - scores[sorted_front[0], obj_idx]
            if obj_range < 1e-12:
                continue

            for k in range(1, len(sorted_front) - 1):
                crowding[sorted_front[k]] += (
                    scores[sorted_front[k + 1], obj_idx] - scores[sorted_front[k - 1], obj_idx]
                ) / obj_range

    return fronts, crowding


def pareto_rank_rewards(scores: np.ndarray, pareto_lambda: float = 0.1) -> np.ndarray:
    """Convert objective scores to scalar rewards via Pareto ranking.

    Higher front index -> lower reward. Within a front, higher crowding -> higher reward.
    """
    n = scores.shape[0]
    if n == 0:
        return np.array([])

    fronts, crowding = nsga2_sort(scores)
    rewards = np.zeros(n)
    n_fronts = len(fronts)

    for front_idx, front in enumerate(fronts):
        # Base reward decreases with front index
        base = 1.0 - (front_idx / max(1, n_fronts))
        for i in front:
            # Bonus for crowding distance (diversity)
            cd = min(crowding[i], 10.0) / 10.0  # Clip and normalize
            rewards[i] = base + pareto_lambda * cd

    return rewards
