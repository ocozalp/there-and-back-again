import torch

from dl.data import Batches, split
from dl.nn import MLP, cross_entropy
from dl.training import SGD, Adam, evaluate, fit


def test_batches_cover_every_example_once():
    x = torch.arange(10).unsqueeze(1).float()
    y = torch.arange(10)
    b = Batches(x, y, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
    assert len(b) == 3
    seen = torch.cat([yb for _, yb in b])
    assert sorted(seen.tolist()) == list(range(10))
    # the pairing between tensors is preserved
    for xb, yb in b:
        assert torch.equal(xb.squeeze(1).long(), yb)


def test_split_is_disjoint_and_complete():
    parts = split(100, (0.7, 0.2, 0.1), torch.Generator().manual_seed(0))
    assert [len(p) for p in parts] == [70, 20, 10]
    assert sorted(torch.cat(parts).tolist()) == list(range(100))


def _lockstep(make_ours, make_theirs, steps=30):
    torch.manual_seed(0)
    p_ours = torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float64))
    p_theirs = torch.nn.Parameter(p_ours.detach().clone())
    ours, theirs = make_ours([p_ours]), make_theirs([p_theirs])
    for _ in range(steps):
        g = torch.randn(5, 3, dtype=torch.float64)
        for p, opt in ((p_ours, ours), (p_theirs, theirs)):
            opt.zero_grad()
            (p * g).sum().backward()
            opt.step()
    return (p_ours - p_theirs).abs().max().item()


def test_sgd_matches_torch_optim():
    gap = _lockstep(lambda p: SGD(p, lr=0.1), lambda p: torch.optim.SGD(p, lr=0.1))
    assert gap < 1e-12
    gap = _lockstep(
        lambda p: SGD(p, lr=0.1, momentum=0.9), lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)
    )
    assert gap < 1e-12


def test_adam_matches_torch_optim():
    gap = _lockstep(lambda p: Adam(p, lr=0.01), lambda p: torch.optim.Adam(p, lr=0.01))
    assert gap < 1e-10


def test_fit_lowers_loss_on_a_toy_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 4, generator=g)
    y = (x[:, 0] + x[:, 1] > 0).long()
    train = Batches(x, y, batch_size=32, shuffle=True, generator=g)
    model = MLP(4, [16], 2)
    device = torch.device("cpu")
    before, _ = evaluate(model, train, cross_entropy, device)
    hist = fit(
        model, train, train, cross_entropy, Adam(model.parameters(), 1e-2), 5, device, log=None
    )
    assert hist.train_loss[-1] < before
    assert hist.val_acc[-1] > 0.9
    assert len(hist.epoch_time) == 5


def test_optimizer_state_is_created_lazily():
    """State must follow the parameter's device, so it cannot be allocated in __init__."""
    p = torch.nn.Parameter(torch.zeros(3))
    opt = Adam([p], lr=0.1)
    assert opt.m == {} and opt.v == {}
    (p * 2).sum().backward()
    opt.step()
    assert set(opt.m) == {id(p)}
