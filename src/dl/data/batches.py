"""Minibatches from in-memory tensors: the hand-written stand-in for ``DataLoader``."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import Tensor


class Batches:
    """Iterate over aligned tensors in chunks of ``batch_size``, reshuffling every epoch.

    All tensors must share their first dimension. Shuffling draws a fresh permutation from the
    generator each time ``__iter__`` is called, so two epochs see different batch compositions and
    a fixed generator seed makes the whole sequence reproducible.
    """

    def __init__(
        self,
        *tensors: Tensor,
        batch_size: int,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ):
        n = tensors[0].shape[0]
        assert all(t.shape[0] == n for t in tensors), "tensors must have the same length"
        self.tensors = tensors
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        if self.shuffle:
            order = torch.randperm(self.n, generator=self.generator)
        else:
            order = torch.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield tuple(t[idx.to(t.device)] for t in self.tensors)


def split(
    n: int, fractions: tuple[float, ...], generator: torch.Generator | None = None
) -> list[Tensor]:
    """Random disjoint index sets covering 0..n-1 in the given proportions."""
    perm = torch.randperm(n, generator=generator)
    sizes = [int(round(f * n)) for f in fractions]
    sizes[-1] = n - sum(sizes[:-1])
    out, start = [], 0
    for s in sizes:
        out.append(perm[start : start + s])
        start += s
    return out
