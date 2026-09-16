"""A scalar automatic-differentiation engine.

Every ``Value`` remembers the values it was computed from and a tiny function that knows how to
push a gradient back to them. ``backward()`` visits the graph in reverse topological order and
calls those functions, which is backpropagation. This is the whole idea behind ``torch.autograd``;
the difference is that PyTorch does it on tensors and in C++.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


class Value:
    __slots__ = ("data", "grad", "label", "_backward", "_prev", "_op")

    def __init__(
        self, data: float, _children: Iterable[Value] = (), _op: str = "", label: str = ""
    ):
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple[Value, ...] = tuple(_children)
        self._op = _op

    # ---- arithmetic ---------------------------------------------------------------------
    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        assert isinstance(exponent, int | float), "only constant exponents are supported"
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            # d(a^n)/da = n * a^(n-1)
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    # ---- nonlinearities -------------------------------------------------------------------
    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d tanh(a)/da = 1 - tanh(a)^2
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            # gradient passes through where the input was positive, is blocked elsewhere
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            # d sigmoid(a)/da = sigmoid(a) * (1 - sigmoid(a))
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def softplus(self) -> Value:
        """log(1 + exp(a)), computed so that neither exp nor log can overflow or hit zero."""
        x = self.data
        # For large positive x, log(1+e^x) = x + log(1+e^-x); for negative x use it directly.
        sp = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
        out = Value(sp, (self,), "softplus")

        def _backward() -> None:
            # d softplus(a)/da = sigmoid(a)
            self.grad += (1.0 / (1.0 + math.exp(-x))) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            # d exp(a)/da = exp(a), which is out.data
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        assert self.data > 0, "log of a non-positive number"
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            # d log(a)/da = 1/a
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ---- derived operators, all expressed through the primitives above --------------------
    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other) -> Value:
        return self + (-other)

    def __truediv__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __radd__(self, other) -> Value:
        return self + other

    def __rmul__(self, other) -> Value:
        return self * other

    def __rsub__(self, other) -> Value:
        return Value(other) + (-self)

    def __rtruediv__(self, other) -> Value:
        return Value(other) * self**-1

    # ---- backpropagation ---------------------------------------------------------------
    def topological_order(self) -> list[Value]:
        """Every node that feeds into this one, ordered so parents come after their children."""
        order: list[Value] = []
        seen: set[int] = set()

        def visit(v: Value) -> None:
            if id(v) in seen:
                return
            seen.add(id(v))
            for child in v._prev:
                visit(child)
            order.append(v)

        visit(self)
        return order

    def backward(self) -> None:
        """Fill ``grad`` on every node in the graph with d(self)/d(node)."""
        order = self.topological_order()
        self.grad = 1.0  # d(self)/d(self)
        for node in reversed(order):
            node._backward()

    def __repr__(self) -> str:
        name = f"{self.label}=" if self.label else ""
        return f"Value({name}{self.data:.4g}, grad={self.grad:.4g})"
