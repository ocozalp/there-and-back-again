"""Losses for a single output. Each returns one ``Value`` averaged over the batch."""

from __future__ import annotations

from collections.abc import Sequence

from .engine import Value


def mse(preds: Sequence[Value], targets: Sequence[float]) -> Value:
    return sum(((p - t) ** 2 for p, t in zip(preds, targets, strict=True)), Value(0.0)) * (
        1.0 / len(preds)
    )


def hinge(scores: Sequence[Value], targets: Sequence[int]) -> Value:
    """Max-margin loss for labels in {-1, +1}: zero once a score has the right sign and margin 1."""
    return sum(((1 - t * s).relu() for s, t in zip(scores, targets, strict=True)), Value(0.0)) * (
        1.0 / len(scores)
    )


def binary_cross_entropy_with_logits(logits: Sequence[Value], targets: Sequence[int]) -> Value:
    """Negative log-likelihood of labels in {0, 1} under p = sigmoid(logit).

    Written as softplus rather than log(sigmoid): -log(p) = softplus(-z) and
    -log(1 - p) = softplus(z). The naive form computes sigmoid first, which rounds to exactly 1.0
    for z above about 37, and log(1 - 1.0) is log(0). Softplus never hits that wall.
    """
    total = Value(0.0)
    for z, t in zip(logits, targets, strict=True):
        total = total + ((-z).softplus() if t == 1 else z.softplus())
    return total * (1.0 / len(logits))


def l2_penalty(params: Sequence[Value], alpha: float) -> Value:
    return sum((p * p for p in params), Value(0.0)) * alpha
