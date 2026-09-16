"""A two-feature toy problem from Bilbo's world, for Phase 1.

Will today's entry be cheerful? Two things about the day are known in advance: how many hours of
daylight there were and how far Bilbo walked. The rule that generates the labels is deliberately
nonlinear so a straight line cannot separate the classes and a small MLP is needed.
"""

from __future__ import annotations

import numpy as np


def cheerful_threshold(distance: np.ndarray) -> np.ndarray:
    """Daylight hours needed for a cheerful entry, as a function of miles walked.

    Bilbo is happiest walking about ten miles; less feels idle and more feels like torture, and
    either way he needs a longer day to stay cheerful.
    """
    return 9.0 + 0.03 * (distance - 10.0) ** 2


def make_mood_dataset(
    n: int = 200, seed: int = 0, noise: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``X`` of shape (n, 2) = [daylight hours, miles walked] and labels in {-1, +1}."""
    rng = np.random.default_rng(seed)
    daylight = rng.uniform(6.0, 16.0, n)
    distance = rng.uniform(0.0, 25.0, n)
    score = daylight - cheerful_threshold(distance) + rng.normal(0.0, noise, n)
    y = np.where(score > 0, 1, -1)
    return np.stack([daylight, distance], axis=1), y


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance features; returns (X_std, mean, std) so new points can be mapped."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, mean, std
