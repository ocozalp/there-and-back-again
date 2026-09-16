"""Gradient checking: compare autograd against finite differences.

If f is a function of a scalar x, then df/dx is approximately (f(x+eps) - f(x-eps)) / (2 eps).
This is slow (two forward passes per parameter) but it does not depend on any backward code, so
it is the ground truth we test the engine against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .engine import Value


def finite_difference_grads(
    f: Callable[[], Value], params: Sequence[Value], eps: float = 1e-6
) -> list[float]:
    grads = []
    for p in params:
        original = p.data
        p.data = original + eps
        plus = f().data
        p.data = original - eps
        minus = f().data
        p.data = original
        grads.append((plus - minus) / (2 * eps))
    return grads


def check_gradients(
    f: Callable[[], Value],
    params: Sequence[Value],
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> tuple[bool, float]:
    """Run backward on f() and compare every parameter's grad with finite differences.

    Returns (all_close, worst_absolute_error).
    """
    for p in params:
        p.grad = 0.0
    out = f()
    out.backward()
    analytic = [p.grad for p in params]
    numeric = finite_difference_grads(f, params, eps)
    worst = 0.0
    ok = True
    for a, n in zip(analytic, numeric, strict=True):
        err = abs(a - n)
        worst = max(worst, err)
        if err > atol + rtol * abs(n):
            ok = False
    return ok, worst
