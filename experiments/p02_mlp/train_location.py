"""Phase 2: the same MLP on Bilbo's diary renders, predicting the location, then every attribute."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch
from common import confusion_matrix, plot_confusion, plot_history, to_numpy_image

from dl.data import Batches, load_diary, split
from dl.nn import MLP, count_parameters, cross_entropy
from dl.training import Adam, evaluate, fit, get_device, save_checkpoint, seed_everything
from dl.viz import image_grid, save_figure
from dl.world import ATTRIBUTES

FIG = Path("docs/02-pytorch-mlp/figures")
OUT = Path("experiments/p02_mlp")
BATCH = 64
EPOCHS = 15

seed_everything(0)
device = get_device()
t0 = time.perf_counter()
data = load_diary(n=8000, size=32, seed=0)
print(
    f"diary data: {len(data)} images {tuple(data.images.shape[1:])}, "
    f"loaded in {time.perf_counter() - t0:.0f}s"
)
g = torch.Generator().manual_seed(0)
tr, va, te = split(len(data), (0.75, 0.125, 0.125), g)


def batches(idx, attr, shuffle):
    return Batches(
        data.images[idx],
        data.labels[attr][idx],
        batch_size=BATCH,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(0),
    )


def run(attr: str, log=None):
    seed_everything(0)
    n_classes = len(ATTRIBUTES[attr])
    m = MLP(3 * 32 * 32, [256], n_classes, activation="relu")
    h = fit(
        m,
        batches(tr, attr, True),
        batches(va, attr, False),
        cross_entropy,
        Adam(m.parameters(), 1e-3),
        EPOCHS,
        device,
        log=log,
    )
    test_loss, test_acc = evaluate(m, batches(te, attr, False), cross_entropy, device)
    return m, h, test_acc


# ---- location: the main run -------------------------------------------------------------
model, hist, test_acc = run("location", log=print)
n_params = count_parameters(model)
print(f"location: {n_params} parameters, test acc {test_acc:.4f}")
save_checkpoint(model, "checkpoints/p02_location_mlp.pt", test_acc=test_acc)
save_figure(
    plot_history(hist, f"Where is Bilbo? MLP on 32x32 renders, {n_params:,} parameters"),
    FIG / "location_curves.png",
)

model.eval()
with torch.no_grad():
    logits = torch.cat([model(x.to(device)).cpu() for x, _ in batches(te, "location", False)])
pred = logits.argmax(-1)
true = data.labels["location"][te]
classes = ATTRIBUTES["location"]
save_figure(
    plot_confusion(
        confusion_matrix(pred, true, len(classes)), classes, "location: test confusion matrix"
    ),
    FIG / "location_confusion.png",
)
show = list(range(12)) + (pred != true).nonzero().squeeze(1)[:6].tolist()
caps = [f"pred {classes[pred[i]]}\ntrue {classes[true[i]]}" for i in show]
save_figure(
    image_grid(
        [to_numpy_image(data.images[te[i]]) for i in show],
        caps,
        ncols=6,
        cell=1.3,
        title="test predictions (last row: the mistakes, if any)",
    ),
    FIG / "location_predictions.png",
)

# ---- every attribute with the same recipe ------------------------------------------------
per_attr = {}
for attr in ATTRIBUTES:
    if attr == "location":
        per_attr[attr] = round(test_acc, 4)
        continue
    _, h, acc = run(attr)
    per_attr[attr] = round(acc, 4)
    print(f"{attr:10s} test acc {acc:.4f}")

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except subprocess.CalledProcessError:
    commit = "uncommitted"
results = {
    "device": str(device),
    "model": "MLP 3072 -> 256 -> n_classes, ReLU",
    "parameters_location": n_params,
    "train/val/test": [len(tr), len(va), len(te)],
    "batch_size": BATCH,
    "epochs": EPOCHS,
    "optimizer": "Adam lr 1e-3 (hand-written)",
    "location_test_acc": round(test_acc, 4),
    "location_final_val_acc": round(hist.val_acc[-1], 4),
    "seconds_per_epoch": round(sum(hist.epoch_time) / len(hist.epoch_time), 2),
    "test_acc_per_attribute": per_attr,
    "chance_per_attribute": {a: round(1 / len(v), 3) for a, v in ATTRIBUTES.items()},
    "commit": commit,
}
(OUT / "results_location.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
