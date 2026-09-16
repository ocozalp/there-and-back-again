"""Shared plotting helpers for the Phase 2 experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from dl.training import History
from dl.viz import PALETTE, use_style

use_style()


def plot_history(hist: History, title: str) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 2.8))
    epochs = np.arange(1, len(hist.train_loss) + 1)
    a1.plot(epochs, hist.train_loss, c=PALETTE["red"], label="train")
    if hist.val_loss:
        a1.plot(epochs, hist.val_loss, c=PALETTE["blue"], label="validation")
    a1.set_xlabel("epoch")
    a1.set_ylabel("cross-entropy")
    a1.set_title("loss")
    a1.legend(fontsize=7)
    if hist.val_acc:
        a2.plot(epochs, [100 * a for a in hist.val_acc], c=PALETTE["green"])
        a2.set_xlabel("epoch")
        a2.set_ylabel("validation accuracy (%)")
        a2.set_title("accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def confusion_matrix(pred: torch.Tensor, true: torch.Tensor, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(pred.tolist(), true.tolist(), strict=True):
        cm[t, p] += 1
    return cm


def plot_confusion(cm: np.ndarray, classes: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                color = "white" if cm[i, j] > cm.max() / 2 else PALETTE["ink"]
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6, color=color)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """(C, H, W) float tensor in [0, 1] -> (H, W) or (H, W, 3) uint8 array for imshow."""
    x = x.detach().cpu()
    if x.shape[0] == 1:
        return (x[0] * 255).round().to(torch.uint8).numpy()
    return (x.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
