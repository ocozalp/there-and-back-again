"""Render a 6x6 grid of random diary scenes with their captions, and one strip per attribute."""

from pathlib import Path

import numpy as np

from dl.viz import image_grid, save_figure
from dl.world import ATTRIBUTES, Scene, generate_examples, render_scene

FIG = Path("docs/00-skeleton/figures")

examples = generate_examples(36, seed=7, size=32)
fig = image_grid([e.image for e in examples], [e.caption for e in examples], ncols=6)
print(save_figure(fig, FIG / "scene_grid.png"))

# one strip per attribute: hold everything else at the default scene and vary one attribute
for name, values in ATTRIBUTES.items():
    imgs = [np.asarray(render_scene(Scene(**{name: v}, seed=1), size=64)) for v in values]
    fig = image_grid(imgs, values, ncols=len(values), cell=1.4, title=f"varying {name}")
    print(save_figure(fig, FIG / f"attribute_{name}.png"))

# the same scene at 32 and 64 pixels
scene = Scene(
    location="forest", weather="rain", time="dusk", companion="dwarf", item="campfire", seed=3
)
imgs = [np.asarray(render_scene(scene, size=s)) for s in (32, 64, 128)]
fig = image_grid(
    imgs, ["32 px", "64 px", "128 px"], ncols=3, cell=2.0, title="the same scene at three sizes"
)
print(save_figure(fig, FIG / "sizes.png"))

with open("experiments/p00_skeleton/sample_entries.md", "w") as f:
    f.write("# Sample captions and diary entries\n\n")
    for e in examples[:8]:
        f.write(f"- **{e.caption}**  \n  {e.entry}\n")
print("wrote experiments/p00_skeleton/sample_entries.md")
