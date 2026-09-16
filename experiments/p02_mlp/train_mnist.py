"""Phase 2: a fully connected classifier on MNIST, with three side experiments.

1. The main run: MLP 784 -> 256 -> 128 -> 10, Adam, 10 epochs, with a held-out validation split.
2. Optimizers compared: plain SGD, SGD with momentum, Adam, same model and epochs.
3. Overfitting on purpose: 500 training images, 150 epochs, train and validation loss diverge.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from common import confusion_matrix, plot_confusion, plot_history, to_numpy_image

from dl.data import MNIST_CLASSES, Batches, load_mnist, split
from dl.nn import MLP, count_parameters, cross_entropy
from dl.training import SGD, Adam, evaluate, fit, get_device, save_checkpoint, seed_everything
from dl.viz import PALETTE, image_grid, save_figure

FIG = Path("docs/02-pytorch-mlp/figures")
OUT = Path("experiments/p02_mlp")
BATCH = 128
EPOCHS = 10

seed_everything(0)
device = get_device()
print("device:", device)

x_all, y_all, x_test, y_test = load_mnist()
g = torch.Generator().manual_seed(0)
train_idx, val_idx = split(len(x_all), (55000 / 60000, 5000 / 60000), g)
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_val, y_val = x_all[val_idx], y_all[val_idx]
print(f"train {len(x_train)}  val {len(x_val)}  test {len(x_test)}")

save_figure(
    image_grid(
        [to_numpy_image(x) for x in x_train[:24]],
        [MNIST_CLASSES[i] for i in y_train[:24]],
        ncols=8,
        cell=1.0,
    ),
    FIG / "mnist_samples.png",
)


def make_batches(x, y, shuffle, seed=0):
    return Batches(
        x, y, batch_size=BATCH, shuffle=shuffle, generator=torch.Generator().manual_seed(seed)
    )


train_b = make_batches(x_train, y_train, True)
val_b = make_batches(x_val, y_val, False)
test_b = make_batches(x_test, y_test, False)

# ---- 1. main run --------------------------------------------------------------------------
model = MLP(28 * 28, [256, 128], 10, activation="relu")
n_params = count_parameters(model)
print(f"parameters: {n_params}")
t0 = time.perf_counter()
hist = fit(model, train_b, val_b, cross_entropy, Adam(model.parameters(), lr=1e-3), EPOCHS, device)
train_time = time.perf_counter() - t0
test_loss, test_acc = evaluate(model, test_b, cross_entropy, device)
print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}  ({train_time:.0f}s)")
save_checkpoint(model, "checkpoints/p02_mnist_mlp.pt", history=hist.as_dict(), test_acc=test_acc)
save_figure(
    plot_history(hist, f"MNIST MLP, {n_params:,} parameters, Adam"), FIG / "mnist_curves.png"
)

# mistakes and confusion matrix on the test set
model.eval()
with torch.no_grad():
    logits = torch.cat([model(x.to(device)).cpu() for x, _ in test_b])
pred = logits.argmax(-1)
wrong = (pred != y_test).nonzero().squeeze(1)
probs = logits.softmax(-1)
caps = [
    f"pred {MNIST_CLASSES[pred[i]]} ({probs[i, pred[i]]:.0%})\ntrue {MNIST_CLASSES[y_test[i]]}"
    for i in wrong[:24]
]
save_figure(
    image_grid(
        [to_numpy_image(x_test[i]) for i in wrong[:24]],
        caps,
        ncols=8,
        cell=1.0,
        title=f"{len(wrong)} test mistakes out of {len(y_test)}",
    ),
    FIG / "mnist_mistakes.png",
)
cm = confusion_matrix(pred, y_test, 10)
save_figure(
    plot_confusion(cm, MNIST_CLASSES, "MNIST test confusion matrix"), FIG / "mnist_confusion.png"
)

# ---- 2. optimizers compared -------------------------------------------------------------
opt_results = {}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 2.8))
configs = [
    ("SGD lr 0.1", lambda p: SGD(p, lr=0.1), PALETTE["blue"]),
    ("SGD + momentum 0.9, lr 0.05", lambda p: SGD(p, lr=0.05, momentum=0.9), PALETTE["green"]),
    ("Adam lr 1e-3", lambda p: Adam(p, lr=1e-3), PALETTE["red"]),
]
for name, make_opt, color in configs:
    seed_everything(0)
    m = MLP(28 * 28, [256, 128], 10, activation="relu")
    h = fit(
        m,
        make_batches(x_train, y_train, True),
        val_b,
        cross_entropy,
        make_opt(m.parameters()),
        5,
        device,
        log=None,
    )
    opt_results[name] = {"val_acc": round(h.val_acc[-1], 4), "val_loss": round(h.val_loss[-1], 4)}
    a1.plot(range(1, 6), h.train_loss, c=color, label=name)
    a2.plot(range(1, 6), [100 * a for a in h.val_acc], c=color, label=name)
    print(f"{name:32s} val acc {h.val_acc[-1]:.4f}")
a1.set_xlabel("epoch")
a1.set_ylabel("train loss")
a1.set_yscale("log")
a2.set_xlabel("epoch")
a2.set_ylabel("validation accuracy (%)")
a1.legend(fontsize=6)
fig.suptitle("same model, three optimizers, five epochs")
fig.tight_layout()
save_figure(fig, FIG / "optimizers.png")

# ---- 3. overfitting on purpose ---------------------------------------------------------
seed_everything(0)
small_n = 500
small_b = make_batches(x_train[:small_n], y_train[:small_n], True)
m = MLP(28 * 28, [256, 128], 10, activation="relu")
h = fit(m, small_b, val_b, cross_entropy, Adam(m.parameters(), lr=1e-3), 150, device, log=None)
fig, ax = plt.subplots(figsize=(4.6, 3))
ax.plot(h.train_loss, c=PALETTE["red"], label="train (500 images)")
ax.plot(h.val_loss, c=PALETTE["blue"], label="validation (5,000 images)")
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy")
ax.set_yscale("log")
ax.set_title("overfitting: the model memorizes 500 images")
ax.legend(fontsize=7)
save_figure(fig, FIG / "overfitting.png")
best_epoch = int(torch.tensor(h.val_loss).argmin()) + 1
print(
    f"overfit run: best val loss at epoch {best_epoch}, "
    f"final train loss {h.train_loss[-1]:.2e}, final val acc {h.val_acc[-1]:.4f}"
)

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except subprocess.CalledProcessError:
    commit = "uncommitted"
results = {
    "device": str(device),
    "model": "MLP 784 -> 256 -> 128 -> 10, ReLU",
    "parameters": n_params,
    "train/val/test": [len(x_train), len(x_val), len(x_test)],
    "batch_size": BATCH,
    "epochs": EPOCHS,
    "optimizer": "Adam lr 1e-3 (hand-written)",
    "final_train_loss": round(hist.train_loss[-1], 4),
    "final_val_acc": round(hist.val_acc[-1], 4),
    "test_acc": round(test_acc, 4),
    "test_mistakes": int(len(wrong)),
    "train_time_seconds": round(train_time, 1),
    "seconds_per_epoch": round(sum(hist.epoch_time) / len(hist.epoch_time), 2),
    "optimizer_comparison_5_epochs": opt_results,
    "overfit_500_images": {
        "best_val_epoch": best_epoch,
        "final_train_loss": f"{h.train_loss[-1]:.2e}",
        "final_val_loss": round(h.val_loss[-1], 4),
        "final_val_acc": round(h.val_acc[-1], 4),
    },
    "commit": commit,
}
(OUT / "results_mnist.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
