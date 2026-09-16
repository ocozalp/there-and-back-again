"""Generate a reproducible set of diary scenes with images and captions."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .captions import diary_entry, short_caption
from .render import render_scene
from .scenario import Scene


@dataclass
class DiaryExample:
    scene: Scene
    image: np.ndarray  # (size, size, 3) uint8
    caption: str
    entry: str


def generate_scenes(n: int, seed: int = 0) -> list[Scene]:
    rng = random.Random(seed)
    return [Scene.random(rng) for _ in range(n)]


def generate_examples(n: int, seed: int = 0, size: int = 32) -> list[DiaryExample]:
    examples = []
    for scene in generate_scenes(n, seed):
        img = np.asarray(render_scene(scene, size), dtype=np.uint8)
        examples.append(DiaryExample(scene, img, short_caption(scene), diary_entry(scene)))
    return examples
