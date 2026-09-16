"""Print the selected device and a seeded tensor. Run twice: the tensor must be identical."""

import torch

from dl.training import get_device, seed_everything

seed_everything(0)
device = get_device()
x = torch.randn(3, device=device)
print(f"device: {device}")
print(f"seeded tensor: {x.tolist()}")
