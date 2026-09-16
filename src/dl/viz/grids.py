"""Image grids with captions."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import use_style


def image_grid(
    images: Sequence[np.ndarray],
    captions: Sequence[str] | None = None,
    ncols: int = 6,
    cell: float = 1.6,
    title: str | None = None,
    wrap: int = 26,
) -> plt.Figure:
    use_style()
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * cell, nrows * (cell + (0.55 if captions else 0)))
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(img, interpolation="nearest")
        if captions is not None:
            ax.set_title("\n".join(textwrap.wrap(captions[i], wrap)), fontsize=6, pad=3)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig
