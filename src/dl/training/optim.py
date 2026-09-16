"""Optimizers written by hand: plain SGD, SGD with momentum, and Adam.

Each holds a list of parameters and updates them in place from ``p.grad``. The interface matches
``torch.optim`` (``zero_grad`` then ``step``) so the train loop does not care which is used.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """p <- p - lr * g, optionally with momentum: v <- mu * v + g, p <- p - lr * v.

    Momentum averages recent gradients, so steps keep going in directions that agree from batch to
    batch and cancel in directions that flip sign.
    """

    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity: dict[int, Tensor] = {}

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if self.momentum > 0:
                # State is created on first use, on the parameter's own device. Creating it in
                # __init__ would pin it to wherever the model lived at that moment (often the CPU,
                # before .to(device)), which is why torch.optim does the same lazy trick.
                v = self.velocity.setdefault(id(p), torch.zeros_like(p))
                v.mul_(self.momentum).add_(p.grad)
                p.sub_(self.lr * v)
            else:
                p.sub_(self.lr * p.grad)


class Adam(Optimizer):
    """Adaptive step sizes: each parameter is scaled by the running size of its own gradient.

    m tracks the mean gradient (momentum), v tracks the mean squared gradient. Dividing m by
    sqrt(v) makes the step roughly lr in every coordinate regardless of gradient scale. The
    bias corrections (1 - beta^t) undo the fact that m and v start at zero.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m: dict[int, Tensor] = {}
        self.v: dict[int, Tensor] = {}

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        bc1 = 1 - self.beta1**self.t
        bc2 = 1 - self.beta2**self.t
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            m = self.m.setdefault(id(p), torch.zeros_like(p))
            v = self.v.setdefault(id(p), torch.zeros_like(p))
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = m / bc1
            v_hat = v / bc2
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))
