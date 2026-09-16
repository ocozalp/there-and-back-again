"""A fully connected layer written out: y = x W^T + b."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """Same math as ``torch.nn.Linear``.

    ``weight`` has shape (n_out, n_in) so that a batch ``x`` of shape (N, n_in) becomes
    ``x @ weight.T`` of shape (N, n_out); the bias of shape (n_out,) broadcasts across the batch.
    Weights start uniform in [-1/sqrt(n_in), 1/sqrt(n_in)], the same scale rule as Phase 1, so each
    output has roughly unit variance for unit-variance inputs. Biases start at zero.
    """

    def __init__(self, n_in: int, n_out: int, generator: torch.Generator | None = None):
        super().__init__()
        bound = 1.0 / math.sqrt(n_in)
        w = torch.empty(n_out, n_in).uniform_(-bound, bound, generator=generator)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias

    def extra_repr(self) -> str:
        return f"n_in={self.weight.shape[1]}, n_out={self.weight.shape[0]}"
