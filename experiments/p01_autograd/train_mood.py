"""Phase 1: train a hand-built MLP to predict whether Bilbo's diary entry will be cheerful.

Everything here runs on the scalar engine in ``dl.autograd``; PyTorch is not imported. Produces the
figures for the chapter and writes results.md with the numbers the chapter quotes.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dl.autograd import MLP, binary_cross_entropy_with_logits, check_gradients
from dl.viz import PALETTE, save_figure, use_style
from dl.world.toy import cheerful_threshold, make_mood_dataset, standardize

FIG = Path("docs/01-autograd/figures")
OUT = Path("experiments/p01_autograd")
N_TRAIN, N_TEST = 150, 200
EPOCHS = 200
LR = 0.5
SWEEP_EPOCHS = 80
SNAPSHOTS = (0, 5, 30, EPOCHS - 1)

use_style()


def forward_all(model: MLP, X: np.ndarray) -> list:
    return [model(list(x))[0] for x in X]


def accuracy(logits, y01) -> float:
    return float(np.mean([(z.data > 0) == (t == 1) for z, t in zip(logits, y01, strict=True)]))


def train(model: MLP, X, y01, lr: float, epochs: int, on_snapshot=None):
    losses, accs = [], []
    for epoch in range(epochs):
        logits = forward_all(model, X)
        loss = binary_cross_entropy_with_logits(logits, y01)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad  # gradient descent: step against the gradient
        losses.append(loss.data)
        accs.append(accuracy(logits, y01))
        if on_snapshot is not None and epoch in SNAPSHOTS:
            on_snapshot(epoch, loss.data)
    return losses, accs


def decision_surface(model: MLP, mean, std, n=50):
    """Model output on a grid over the raw feature space (daylight, distance)."""
    d = np.linspace(6, 16, n)
    m = np.linspace(0, 25, n)
    D, M = np.meshgrid(d, m)
    pts = (np.stack([D.ravel(), M.ravel()], 1) - mean) / std
    Z = np.array([z.data for z in forward_all(model, pts)]).reshape(D.shape)
    return D, M, Z


def plot_data(ax, X, y, title=None):
    cheer = y == 1
    ax.scatter(
        X[cheer, 0], X[cheer, 1], s=12, c=PALETTE["gold"], label="cheerful", edgecolors="none"
    )
    ax.scatter(
        X[~cheer, 0], X[~cheer, 1], s=12, c=PALETTE["blue"], label="gloomy", edgecolors="none"
    )
    m = np.linspace(0, 25, 100)
    ax.plot(cheerful_threshold(m), m, "--", c=PALETTE["ink"], lw=0.8, label="true rule")
    ax.set_xlim(6, 16)
    ax.set_ylim(0, 25)
    ax.set_xlabel("hours of daylight")
    ax.set_ylabel("miles walked")
    if title:
        ax.set_title(title)


def main() -> None:
    X_raw, y = make_mood_dataset(N_TRAIN, seed=0)
    X, mean, std = standardize(X_raw)
    Xt_raw, yt = make_mood_dataset(N_TEST, seed=1)
    Xt = (Xt_raw - mean) / std
    y01 = [1 if t == 1 else 0 for t in y]
    yt01 = [1 if t == 1 else 0 for t in yt]

    # Figure: the dataset and the rule that made it
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    plot_data(ax, X_raw, y, "Will today's entry be cheerful?")
    ax.legend(loc="upper left", fontsize=7)
    save_figure(fig, FIG / "mood_dataset.png")

    # Main run, with decision-boundary snapshots
    model = MLP(2, [8, 8, 1], nonlin="tanh", seed=0)
    n_params = len(model.parameters())
    snaps: list[tuple[int, float, np.ndarray]] = []

    def on_snapshot(epoch, loss):
        _, _, Z = decision_surface(model, mean, std)
        snaps.append((epoch, loss, Z))

    t0 = time.perf_counter()
    losses, accs = train(model, X, y01, LR, EPOCHS, on_snapshot)
    train_time = time.perf_counter() - t0

    final_train_acc = accuracy(forward_all(model, X), y01)
    test_acc = accuracy(forward_all(model, Xt), yt01)

    # Gradient check on the trained model, against finite differences
    def f():
        return binary_cross_entropy_with_logits(forward_all(model, X[:20]), y01[:20])

    ok, worst = check_gradients(f, model.parameters(), atol=1e-6, rtol=1e-4)

    # Figure: decision boundary over training
    D, M, _ = decision_surface(model, mean, std)
    fig, axes = plt.subplots(1, len(snaps), figsize=(3.2 * len(snaps), 3.2), sharey=True)
    for ax, (epoch, loss, Z) in zip(axes, snaps, strict=True):
        ax.contourf(
            D, M, Z, levels=[-50, 0, 50], colors=[PALETTE["blue"], PALETTE["gold"]], alpha=0.25
        )
        ax.contour(D, M, Z, levels=[0], colors=[PALETTE["red"]], linewidths=1.2)
        plot_data(ax, X_raw, y, f"epoch {epoch}, loss {loss:.3f}")
        ax.grid(False)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, FIG / "decision_boundary_evolution.png")

    # Figure: training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.8))
    a1.plot(losses, c=PALETTE["red"])
    a1.set_xlabel("epoch")
    a1.set_ylabel("binary cross-entropy")
    a1.set_title("loss")
    a2.plot(accs, c=PALETTE["green"])
    a2.set_xlabel("epoch")
    a2.set_ylabel("training accuracy")
    a2.set_ylim(0.4, 1.02)
    a2.set_title("accuracy")
    fig.tight_layout()
    save_figure(fig, FIG / "training_curves.png")

    # Learning-rate sweep, same initialization each time
    sweep = {}
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for lr, color in ((0.05, PALETTE["blue"]), (0.5, PALETTE["green"]), (10.0, PALETTE["red"])):
        m = MLP(2, [8, 8, 1], nonlin="tanh", seed=0)
        curve, _ = train(m, X, y01, lr, SWEEP_EPOCHS)
        sweep[lr] = curve[-1]
        ax.plot(curve, c=color, label=f"lr = {lr}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("learning rate: too small, about right, too large")
    ax.legend(fontsize=7)
    save_figure(fig, FIG / "learning_rate_sweep.png")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"

    results = {
        "model": "MLP(2 -> 8 -> 8 -> 1), tanh",
        "parameters": n_params,
        "train_points": N_TRAIN,
        "test_points": N_TEST,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "optimizer": "full-batch gradient descent",
        "loss": "binary cross-entropy with logits",
        "final_train_loss": round(losses[-1], 4),
        "final_train_accuracy": round(final_train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_time_seconds": round(train_time, 1),
        "gradient_check_passed": ok,
        "gradient_check_worst_abs_error": worst,
        "lr_sweep_final_loss": {str(k): round(v, 4) for k, v in sweep.items()},
        "commit": commit,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Phase 1 results",
        "",
        "Generated by `train_mood.py`. The chapter quotes these numbers.",
        "",
    ]
    lines += [f"- **{k}**: {v}" for k, v in results.items()]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
