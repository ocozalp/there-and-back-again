import math
import random

import pytest
import torch

from dl.autograd import MLP, Value, binary_cross_entropy_with_logits, check_gradients, hinge, mse


def test_simple_expression_forward_and_backward():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    d = a * b + c  # 4
    L = d.tanh()
    L.backward()
    assert math.isclose(d.data, 4.0)
    t = math.tanh(4.0)
    assert math.isclose(L.grad, 1.0)
    assert math.isclose(d.grad, 1 - t * t, rel_tol=1e-9)
    assert math.isclose(a.grad, (1 - t * t) * b.data, rel_tol=1e-9)
    assert math.isclose(b.grad, (1 - t * t) * a.data, rel_tol=1e-9)
    assert math.isclose(c.grad, 1 - t * t, rel_tol=1e-9)


def test_gradient_accumulates_when_a_value_is_used_twice():
    a = Value(3.0)
    b = a * a  # da/da contributes twice: d(a*a)/da = 2a
    b.backward()
    assert math.isclose(a.grad, 6.0)


@pytest.mark.parametrize("seed", range(5))
def test_random_graphs_match_finite_differences(seed):
    rng = random.Random(seed)
    xs = [Value(rng.uniform(0.5, 2.0), label=f"x{i}") for i in range(4)]

    def f():
        a = xs[0] * xs[1] + xs[2] ** 2 - xs[3] / xs[0]
        b = a.tanh() * xs[1].exp() + (xs[2] + 1.5).log()
        c = (b * 0.5 - xs[3]).relu() + b.sigmoid()
        return c * c + 3 - 2 / (xs[0] + 1)

    ok, worst = check_gradients(f, xs, atol=1e-6, rtol=1e-4)
    assert ok, f"worst error {worst}"
    assert worst < 1e-4


def test_matches_torch_autograd():
    a = Value(0.7)
    b = Value(-1.3)
    c = Value(2.1)
    out = ((a * b).tanh() + c.exp() * a - (b**2 + 1).log()) / (c + a.relu())
    out.backward()

    ta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(-1.3, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(2.1, dtype=torch.float64, requires_grad=True)
    tout = ((ta * tb).tanh() + tc.exp() * ta - (tb**2 + 1).log()) / (tc + ta.relu())
    tout.backward()

    assert math.isclose(out.data, tout.item(), rel_tol=1e-12)
    for v, t in ((a, ta), (b, tb), (c, tc)):
        assert math.isclose(v.grad, t.grad.item(), rel_tol=1e-9, abs_tol=1e-12)


def test_mlp_parameter_count_and_shapes():
    model = MLP(2, [8, 8, 1])
    # (2*8 + 8) + (8*8 + 8) + (8*1 + 1) = 24 + 72 + 9 = 105
    assert len(model.parameters()) == 105
    out = model([0.5, -0.5])
    assert len(out) == 1
    assert isinstance(out[0], Value)


def test_mlp_gradients_match_finite_differences():
    model = MLP(2, [4, 1], seed=1)
    xs = [[0.3, -1.2], [1.5, 0.4], [-0.7, 0.9]]
    ys = [1, -1, 1]

    def f():
        scores = [model(x)[0] for x in xs]
        return hinge(scores, ys) + mse(scores, [float(y) for y in ys]) * 0.1

    ok, worst = check_gradients(f, model.parameters(), atol=1e-6, rtol=1e-4)
    assert ok, f"worst error {worst}"


def test_bce_matches_torch():
    logits = [Value(0.3), Value(-1.7), Value(2.2)]
    targets = [1, 0, 1]
    loss = binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    tl = torch.tensor([0.3, -1.7, 2.2], dtype=torch.float64, requires_grad=True)
    tt = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    tloss = torch.nn.functional.binary_cross_entropy_with_logits(tl, tt)
    tloss.backward()
    assert math.isclose(loss.data, tloss.item(), rel_tol=1e-9)
    for v, g in zip(logits, tl.grad.tolist(), strict=True):
        assert math.isclose(v.grad, g, rel_tol=1e-9, abs_tol=1e-12)


def test_zero_grad():
    model = MLP(2, [3, 1])
    model([1.0, 2.0])[0].backward()
    assert any(p.grad != 0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())


def test_bce_is_stable_for_saturated_logits():
    logits = [Value(60.0), Value(-60.0), Value(60.0)]
    targets = [1, 0, 0]
    loss = binary_cross_entropy_with_logits(logits, targets)  # naive log(sigmoid) would hit log(0)
    loss.backward()
    tl = torch.tensor([60.0, -60.0, 60.0], dtype=torch.float64, requires_grad=True)
    tt = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    tloss = torch.nn.functional.binary_cross_entropy_with_logits(tl, tt)
    tloss.backward()
    assert math.isclose(loss.data, tloss.item(), rel_tol=1e-9)
    for v, g in zip(logits, tl.grad.tolist(), strict=True):
        assert math.isclose(v.grad, g, rel_tol=1e-9, abs_tol=1e-12)


def test_softplus_matches_torch():
    for x in (-40.0, -2.0, 0.0, 3.0, 40.0):
        v = Value(x)
        out = v.softplus()
        out.backward()
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        tout = torch.nn.functional.softplus(t)
        tout.backward()
        assert math.isclose(out.data, tout.item(), rel_tol=1e-12, abs_tol=1e-300)
        assert math.isclose(v.grad, t.grad.item(), rel_tol=1e-9, abs_tol=1e-300)
