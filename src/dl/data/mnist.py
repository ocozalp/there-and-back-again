"""MNIST as plain tensors: 70,000 handwritten digits, 28x28 grayscale, ten classes."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torchvision.datasets import MNIST

MNIST_CLASSES = [str(i) for i in range(10)]


def load_mnist(root: str | Path = "data") -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (X_train, y_train, X_test, y_test); images are float32 in [0, 1], (N, 1, 28, 28)."""
    train = MNIST(str(root), train=True, download=True)
    test = MNIST(str(root), train=False, download=True)
    x_train = train.data.unsqueeze(1).float() / 255.0
    x_test = test.data.unsqueeze(1).float() / 255.0
    return x_train, train.targets.clone(), x_test, test.targets.clone()
