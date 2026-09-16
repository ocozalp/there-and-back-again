"""Neurons, layers, and an MLP built from ``Value``s. Same shape as ``torch.nn``, scaled down."""

from __future__ import annotations

import random

from .engine import Value


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """A weighted sum of inputs plus a bias, followed by an optional nonlinearity."""

    def __init__(self, n_in: int, nonlin: str | None = "tanh", rng: random.Random | None = None):
        rng = rng or random.Random(0)
        scale = n_in**-0.5  # keeps pre-activations near unit scale regardless of fan-in
        self.w = [Value(rng.uniform(-1, 1) * scale, label="w") for _ in range(n_in)]
        self.b = Value(0.0, label="b")
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        return act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(
        self, n_in: int, n_out: int, nonlin: str | None = "tanh", rng: random.Random | None = None
    ):
        self.neurons = [Neuron(n_in, nonlin, rng) for _ in range(n_out)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Layers stacked; the last layer is linear so it can output any real number."""

    def __init__(self, n_in: int, sizes: list[int], nonlin: str = "tanh", seed: int = 0):
        rng = random.Random(seed)
        dims = [n_in] + sizes
        self.layers = [
            Layer(dims[i], dims[i + 1], nonlin if i < len(sizes) - 1 else None, rng)
            for i in range(len(sizes))
        ]

    def __call__(self, x: list[Value] | list[float]) -> list[Value]:
        h: list[Value] = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            h = layer(h)
        return h

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        sizes = [len(layer.neurons) for layer in self.layers]
        return f"MLP({len(self.parameters())} parameters, layers={sizes})"
