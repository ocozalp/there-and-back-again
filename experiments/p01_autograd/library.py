"""The library equivalent of Phase 1: torch.autograd.

Three demonstrations:
1. The same scalar expression in our engine and in PyTorch gives the same gradients.
2. PyTorch's graph is visible through grad_fn, and torch.autograd.gradcheck is our check_gradients.
3. An nn.Sequential initialized with our MLP's exact weights and trained with the same full-batch
   gradient descent follows the same loss curve to within floating-point noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dl.autograd import MLP, Value, binary_cross_entropy_with_logits
from dl.viz import PALETTE, save_figure, use_style
from dl.world.toy import make_mood_dataset, standardize

FIG = Path("docs/01-autograd/figures")
use_style()
torch.set_default_dtype(torch.float64)

print("== 1. same expression, two engines ==")
a, b, c = Value(0.7, label="a"), Value(-1.3, label="b"), Value(2.1, label="c")
ours = ((a * b).tanh() + c.exp() * a - (b**2 + 1).log()) / (c + a.relu())
ours.backward()

ta = torch.tensor(0.7, requires_grad=True)
tb = torch.tensor(-1.3, requires_grad=True)
tc = torch.tensor(2.1, requires_grad=True)
theirs = ((ta * tb).tanh() + tc.exp() * ta - (tb**2 + 1).log()) / (tc + ta.relu())
theirs.backward()
print(f"value   ours={ours.data:.12f}  torch={theirs.item():.12f}")
for name, v, t in (("a", a, ta), ("b", b, tb), ("c", c, tc)):
    print(f"d/d{name}    ours={v.grad:.12f}  torch={t.grad.item():.12f}")
    assert math.isclose(v.grad, t.grad.item(), rel_tol=1e-9)

print("\n== 2. torch keeps the same graph: walk grad_fn like our _prev ==")
node = theirs.grad_fn
depth = 0
while node is not None and depth < 6:
    print("  " * depth + type(node).__name__)
    node = node.next_functions[0][0] if node.next_functions else None
    depth += 1
print("(each *Backward object is our _backward closure plus a pointer to its inputs)")

print("\n== 2b. torch.autograd.gradcheck is our check_gradients ==")


def f(x, y):
    return (x * y).tanh() + x.exp() / (y**2 + 1)


ok = torch.autograd.gradcheck(
    f, (torch.tensor(0.4, requires_grad=True), torch.tensor(-0.9, requires_grad=True))
)
print(f"gradcheck passed: {ok}")

print("\n== 3. our MLP vs nn.Sequential with identical weights and identical training ==")
X_raw, y = make_mood_dataset(150, seed=0)
X, _, _ = standardize(X_raw)
y01 = [1 if t == 1 else 0 for t in y]
ours_mlp = MLP(2, [8, 8, 1], nonlin="tanh", seed=0)

torch_mlp = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 1),
)
with torch.no_grad():
    for layer, lin in zip(
        ours_mlp.layers, [m for m in torch_mlp if isinstance(m, torch.nn.Linear)], strict=True
    ):
        lin.weight.copy_(torch.tensor([[w.data for w in n.w] for n in layer.neurons]))
        lin.bias.copy_(torch.tensor([n.b.data for n in layer.neurons]))
n_ours = len(ours_mlp.parameters())
n_torch = sum(p.numel() for p in torch_mlp.parameters())
print(f"parameters: ours={n_ours} torch={n_torch}")

Xt = torch.tensor(X)
yt = torch.tensor(y01, dtype=torch.float64)
lr, epochs = 0.5, 60
loss_ours, loss_torch = [], []
for _ in range(epochs):
    logits = [ours_mlp(list(x))[0] for x in X]
    loss = binary_cross_entropy_with_logits(logits, y01)
    ours_mlp.zero_grad()
    loss.backward()
    for p in ours_mlp.parameters():
        p.data -= lr * p.grad
    loss_ours.append(loss.data)

    out = torch_mlp(Xt).squeeze(1)
    tloss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt)
    torch_mlp.zero_grad()
    tloss.backward()
    with torch.no_grad():
        for p in torch_mlp.parameters():
            p -= lr * p.grad
    loss_torch.append(tloss.item())

gap = float(np.max(np.abs(np.array(loss_ours) - np.array(loss_torch))))
print(f"final loss: ours={loss_ours[-1]:.10f} torch={loss_torch[-1]:.10f}")
print(f"largest gap between the two loss curves over {epochs} epochs: {gap:.2e}")

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.plot(loss_ours, c=PALETTE["green"], lw=2.5, label="our engine")
ax.plot(loss_torch, c=PALETTE["gold"], lw=1, ls="--", label="torch.autograd")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title(f"same weights, same steps: curves differ by at most {gap:.1e}")
ax.legend(fontsize=7)
save_figure(fig, FIG / "ours_vs_torch.png")
print(f"wrote {FIG / 'ours_vs_torch.png'}")
