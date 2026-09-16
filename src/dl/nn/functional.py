"""Hand-written versions of the functions a classifier needs, on tensors.

These mirror ``torch.nn.functional`` and are compared against it in the Phase 2 tests. Later phases
use the library versions; these exist so the math is written out once.
"""

from __future__ import annotations

from torch import Tensor


def log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """log(softmax(z)) computed as z - logsumexp(z), with the max subtracted first for stability.

    softmax(z)_i = exp(z_i) / sum_j exp(z_j). Taking the log gives z_i - log(sum_j exp(z_j)).
    Subtracting m = max(z) inside both exp and log changes nothing mathematically but keeps
    exp() from overflowing: exp(z - m) is at most 1.
    """
    m = logits.max(dim=dim, keepdim=True).values
    shifted = logits - m
    return shifted - shifted.exp().sum(dim=dim, keepdim=True).log()


def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    return log_softmax(logits, dim).exp()


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean negative log-probability of the correct class.

    ``logits`` is (N, C); ``targets`` is (N,) of class indices. For each row we pick out the
    log-probability of the target class and negate it.
    """
    logp = log_softmax(logits, dim=-1)
    picked = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -picked.mean()


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()
