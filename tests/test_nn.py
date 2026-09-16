import torch
import torch.nn.functional as F

from dl.autograd import MLP as ScalarMLP
from dl.autograd import binary_cross_entropy_with_logits
from dl.nn import MLP, Linear, accuracy, cross_entropy, log_softmax, softmax


def test_linear_matches_torch_functional():
    g = torch.Generator().manual_seed(0)
    layer = Linear(5, 3, generator=g)
    x = torch.randn(7, 5)
    assert layer(x).shape == (7, 3)
    assert torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias))


def test_softmax_and_log_softmax_match_torch():
    z = torch.randn(4, 6, dtype=torch.float64)
    assert torch.allclose(softmax(z), F.softmax(z, dim=-1))
    assert torch.allclose(log_softmax(z), F.log_softmax(z, dim=-1))
    assert torch.allclose(softmax(z).sum(-1), torch.ones(4, dtype=torch.float64))


def test_log_softmax_is_stable_for_huge_logits():
    z = torch.tensor([[1000.0, 0.0, -1000.0]], dtype=torch.float64)
    out = log_softmax(z)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, F.log_softmax(z, dim=-1))


def test_cross_entropy_matches_torch_including_gradient():
    z = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 5, (8,))
    ours = cross_entropy(z, y)
    ours.backward()
    g_ours = z.grad.clone()
    z.grad = None
    theirs = F.cross_entropy(z, y)
    theirs.backward()
    assert torch.allclose(ours, theirs)
    assert torch.allclose(g_ours, z.grad)
    # the famous gradient: softmax(z) - onehot(y), divided by N for the mean
    expected = (softmax(z.detach()) - F.one_hot(y, 5).to(torch.float64)) / 8
    assert torch.allclose(g_ours, expected)


def test_mlp_flattens_and_shapes():
    model = MLP(28 * 28, [32], 10)
    x = torch.randn(3, 1, 28, 28)
    assert model(x).shape == (3, 10)
    assert accuracy(model(x), torch.tensor([0, 1, 2])) in {0.0, 1 / 3, 2 / 3, 1.0}


def test_tensor_mlp_agrees_with_scalar_engine():
    """The exit criterion of Phase 2: the two engines agree on loss and every gradient."""
    scalar = ScalarMLP(2, [4, 4, 1], nonlin="tanh", seed=3)
    tensor = MLP(2, [4, 4], 1, activation="tanh").double()
    with torch.no_grad():
        for s_layer, t_layer in zip(scalar.layers, tensor.layers, strict=True):
            t_layer.weight.copy_(
                torch.tensor([[w.data for w in n.w] for n in s_layer.neurons], dtype=torch.float64)
            )
            t_layer.bias.copy_(
                torch.tensor([n.b.data for n in s_layer.neurons], dtype=torch.float64)
            )

    xs = [[0.3, -1.2], [1.5, 0.4], [-0.7, 0.9], [0.1, 0.2]]
    ys = [1, 0, 1, 0]
    loss_s = binary_cross_entropy_with_logits([scalar(x)[0] for x in xs], ys)
    scalar.zero_grad()
    loss_s.backward()

    logits = tensor(torch.tensor(xs, dtype=torch.float64)).squeeze(1)
    loss_t = F.binary_cross_entropy_with_logits(logits, torch.tensor(ys, dtype=torch.float64))
    loss_t.backward()

    assert abs(loss_s.data - loss_t.item()) < 1e-6
    worst = 0.0
    for s_layer, t_layer in zip(scalar.layers, tensor.layers, strict=True):
        gw = torch.tensor([[w.grad for w in n.w] for n in s_layer.neurons], dtype=torch.float64)
        gb = torch.tensor([n.b.grad for n in s_layer.neurons], dtype=torch.float64)
        worst = max(worst, (gw - t_layer.weight.grad).abs().max().item())
        worst = max(worst, (gb - t_layer.bias.grad).abs().max().item())
    assert worst < 1e-6
