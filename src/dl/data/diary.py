"""Bilbo's diary renders as tensors, cached on disk so experiments start instantly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from dl.world import ATTRIBUTES, generate_scenes, render_scene, short_caption


@dataclass
class DiaryTensors:
    images: Tensor  # (N, 3, size, size) float32 in [0, 1]
    labels: dict[str, Tensor]  # attribute name -> (N,) int64
    captions: list[str]

    def __len__(self) -> int:
        return self.images.shape[0]


def load_diary(
    n: int = 8000, size: int = 32, seed: int = 0, root: str | Path = "data"
) -> DiaryTensors:
    path = Path(root) / f"diary_n{n}_s{size}_seed{seed}.pt"
    if path.exists():
        blob = torch.load(path)
        return DiaryTensors(blob["images"], blob["labels"], blob["captions"])
    scenes = generate_scenes(n, seed)
    imgs = np.stack([np.asarray(render_scene(s, size), dtype=np.uint8) for s in scenes])
    images = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    labels = {
        name: torch.tensor([s.labels()[name] for s in scenes], dtype=torch.int64)
        for name in ATTRIBUTES
    }
    captions = [short_caption(s) for s in scenes]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels, "captions": captions}, path)
    return DiaryTensors(images, labels, captions)
