import torch

from dl.training import get_device, seed_everything


def test_get_device_returns_device():
    assert isinstance(get_device(), torch.device)
    assert get_device("cpu").type == "cpu"


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.equal(a, b)
