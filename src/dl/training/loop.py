"""The generic train loop every later phase uses.

One epoch: for each batch, forward, loss, zero_grad, backward, step. After each epoch, evaluate
on the validation set. Nothing here knows what the model or the data is.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .optim import Optimizer

LossFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: [round(x, 5) for x in v] for k, v in self.__dict__.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, batches: Iterable, loss_fn: LossFn, device: torch.device
) -> tuple[float, float]:
    """Mean loss and accuracy over every batch. Uses eval mode and records no graph."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(-1) == y).sum().item()
        n += len(y)
    model.train()
    return total_loss / n, correct / n


def fit(
    model: nn.Module,
    train_batches: Iterable,
    val_batches: Iterable | None,
    loss_fn: LossFn,
    optimizer: Optimizer | torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    log: Callable[[str], None] | None = print,
    on_epoch: Callable[[int, nn.Module, History], None] | None = None,
) -> History:
    model.to(device)
    model.train()
    history = History()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        running, n = 0.0, 0
        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            n += len(y)
        if device.type == "mps":
            torch.mps.synchronize()
        history.train_loss.append(running / n)
        history.epoch_time.append(time.perf_counter() - t0)
        msg = f"epoch {epoch + 1:3d}  train loss {running / n:.4f}"
        if val_batches is not None:
            vl, va = evaluate(model, val_batches, loss_fn, device)
            history.val_loss.append(vl)
            history.val_acc.append(va)
            msg += f"  val loss {vl:.4f}  val acc {va:.4f}"
        msg += f"  ({history.epoch_time[-1]:.1f}s)"
        if log:
            log(msg)
        if on_epoch:
            on_epoch(epoch, model, history)
    return history


def save_checkpoint(model: nn.Module, path: str | Path, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)
    return path


def load_checkpoint(model: nn.Module, path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return ckpt
