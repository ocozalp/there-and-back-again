"""A multilayer perceptron on tensors: the Phase 1 MLP, one matrix multiply per layer."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .linear import Linear

ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {"relu": torch.relu, "tanh": torch.tanh}


class MLP(nn.Module):
    """``n_in -> hidden[0] -> ... -> n_out``; the activation sits between layers, not after the
    last.

    Inputs of any shape are flattened to (N, n_in) first, so an image batch (N, 1, 28, 28) is fine.
    """

    def __init__(
        self,
        n_in: int,
        hidden: list[int],
        n_out: int,
        activation: str = "relu",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        dims = [n_in, *hidden, n_out]
        self.layers = nn.ModuleList(
            [Linear(dims[i], dims[i + 1], generator) for i in range(len(dims) - 1)]
        )
        self.activation = activation
        self.act = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
