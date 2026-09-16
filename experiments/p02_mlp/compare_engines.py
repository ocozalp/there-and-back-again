"""Exit criterion: the Phase 1 scalar engine and the tensor MLP agree on loss and every gradient."""

import torch
import torch.nn.functional as F

from dl.autograd import MLP as ScalarMLP
from dl.autograd import binary_cross_entropy_with_logits
from dl.nn import MLP
from dl.world.toy import make_mood_dataset, standardize

scalar = ScalarMLP(2, [8, 8, 1], nonlin="tanh", seed=0)
tensor = MLP(2, [8, 8], 1, activation="tanh").double()
with torch.no_grad():
    for s_layer, t_layer in zip(scalar.layers, tensor.layers, strict=True):
        t_layer.weight.copy_(
            torch.tensor([[w.data for w in n.w] for n in s_layer.neurons], dtype=torch.float64)
        )
        t_layer.bias.copy_(torch.tensor([n.b.data for n in s_layer.neurons], dtype=torch.float64))

X_raw, y = make_mood_dataset(150, seed=0)
X, _, _ = standardize(X_raw)
y01 = [1 if t == 1 else 0 for t in y]

loss_s = binary_cross_entropy_with_logits([scalar(list(x))[0] for x in X], y01)
scalar.zero_grad()
loss_s.backward()

logits = tensor(torch.tensor(X)).squeeze(1)
loss_t = F.binary_cross_entropy_with_logits(logits, torch.tensor(y01, dtype=torch.float64))
loss_t.backward()

worst = 0.0
for s_layer, t_layer in zip(scalar.layers, tensor.layers, strict=True):
    gw = torch.tensor([[w.grad for w in n.w] for n in s_layer.neurons], dtype=torch.float64)
    gb = torch.tensor([n.b.grad for n in s_layer.neurons], dtype=torch.float64)
    worst = max(worst, (gw - t_layer.weight.grad).abs().max().item())
    worst = max(worst, (gb - t_layer.bias.grad).abs().max().item())

print(f"loss   scalar engine={loss_s.data:.15f}  tensor={loss_t.item():.15f}")
print(f"loss gap {abs(loss_s.data - loss_t.item()):.2e}")
print(f"largest gradient gap over 105 parameters: {worst:.2e}")
print("scalar nodes created for one forward pass of 150 examples:", len(loss_s.topological_order()))
print("tensor nodes for the same forward pass: about 10 (one per matrix op)")
