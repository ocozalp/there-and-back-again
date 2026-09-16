"""The library equivalents of Phase 2: torch.nn, torch.optim, torch.utils.data.

1. Our functional pieces against torch.nn.functional.
2. Our Adam against torch.optim.Adam in lockstep on a real model.
3. Our Batches against DataLoader.
4. The MNIST classifier built entirely from library parts, trained with the same recipe.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl.data import Batches, load_mnist, split
from dl.nn import MLP, cross_entropy, log_softmax, softmax
from dl.training import Adam, evaluate, fit, get_device, seed_everything

device = get_device()
seed_everything(0)

print("== 1. functional pieces ==")
z = torch.randn(6, 10, dtype=torch.float64)
y = torch.randint(0, 10, (6,))
print(f"softmax      max gap {(softmax(z) - F.softmax(z, -1)).abs().max():.1e}")
print(f"log_softmax  max gap {(log_softmax(z) - F.log_softmax(z, -1)).abs().max():.1e}")
print(f"cross_entropy gap    {(cross_entropy(z, y) - F.cross_entropy(z, y)).abs():.1e}")
print(
    "nn.CrossEntropyLoss()(z, y) is the same function as a module:",
    f"{(nn.CrossEntropyLoss()(z, y) - cross_entropy(z, y)).abs():.1e}",
)

print("\n== 2. our Adam vs torch.optim.Adam, same model, same batches ==")
x_all, y_all, x_test, y_test = load_mnist()
tr, va = split(len(x_all), (55000 / 60000, 5000 / 60000), torch.Generator().manual_seed(0))
seed_everything(0)
ours_model = MLP(784, [64], 10)
theirs_model = MLP(784, [64], 10)
theirs_model.load_state_dict(ours_model.state_dict())
ours_opt = Adam(ours_model.parameters(), lr=1e-3)
theirs_opt = torch.optim.Adam(theirs_model.parameters(), lr=1e-3)
xb, yb = x_all[tr[:2048]], y_all[tr[:2048]]
for i in range(20):
    x, t = xb[i * 64 : (i + 1) * 64], yb[i * 64 : (i + 1) * 64]
    for m, o in ((ours_model, ours_opt), (theirs_model, theirs_opt)):
        o.zero_grad()
        F.cross_entropy(m(x), t).backward()
        o.step()
gap = max(
    (a - b).abs().max().item()
    for a, b in zip(ours_model.parameters(), theirs_model.parameters(), strict=True)
)
print(f"largest parameter gap after 20 Adam steps: {gap:.1e}")

print("\n== 3. Batches vs DataLoader ==")
ds = TensorDataset(x_all[tr], y_all[tr])
loader = DataLoader(ds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(0))
ours = Batches(
    x_all[tr], y_all[tr], batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(0)
)
xb_theirs, yb_theirs = next(iter(loader))
xb_ours, yb_ours = next(iter(ours))
print(
    f"DataLoader batch: x {tuple(xb_theirs.shape)} y {tuple(yb_theirs.shape)};"
    f"  ours: x {tuple(xb_ours.shape)} y {tuple(yb_ours.shape)}"
)
print(f"batches per epoch: DataLoader {len(loader)}, ours {len(ours)}")
print(
    "DataLoader adds: worker processes, pinned memory, custom collate, samplers, iterable datasets."
)

print("\n== 4. the MNIST classifier from library parts only ==")
seed_everything(0)
lib_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
lib_loss = nn.CrossEntropyLoss()
lib_opt = torch.optim.Adam(lib_model.parameters(), lr=1e-3)
val_loader = DataLoader(TensorDataset(x_all[va], y_all[va]), batch_size=128)
t0 = time.perf_counter()
hist = fit(lib_model, loader, val_loader, lib_loss, lib_opt, 5, device, log=None)
test_loss, test_acc = evaluate(
    lib_model, DataLoader(TensorDataset(x_test, y_test), batch_size=128), lib_loss, device
)
print(
    "nn.Sequential + nn.CrossEntropyLoss + optim.Adam + DataLoader, 5 epochs: "
    f"test acc {test_acc:.4f} ({time.perf_counter() - t0:.0f}s)"
)
print("(nn.Linear initializes its bias uniformly instead of at zero; otherwise identical to ours)")
print("\nparameter names in the library model:")
for name, p in lib_model.named_parameters():
    print(f"  {name:12s} {tuple(p.shape)}")
