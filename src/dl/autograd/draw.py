"""Draw a computation graph with matplotlib (no graphviz needed).

Nodes are laid out in columns by depth: leaves on the left, the output on the right. Each value
box shows its label, data, and grad; each small circle is the operation that produced a value.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dl.viz.style import PALETTE, use_style

from .engine import Value


def _depths(root: Value) -> dict[int, int]:
    """Longest path from a leaf to each node, so the layout reads left to right."""
    depth: dict[int, int] = {}
    for v in root.topological_order():
        depth[id(v)] = 0 if not v._prev else 1 + max(depth[id(c)] for c in v._prev)
    return depth


def draw_graph(root: Value, title: str | None = None, show_grad: bool = True) -> plt.Figure:
    use_style()
    nodes = root.topological_order()
    depth = _depths(root)
    columns: dict[int, list[Value]] = defaultdict(list)
    for v in nodes:
        columns[depth[id(v)]].append(v)
    ncols = max(columns) + 1
    nrows = max(len(col) for col in columns.values())

    pos: dict[int, tuple[float, float]] = {}
    for d, col in columns.items():
        for i, v in enumerate(col):
            y = (nrows - 1) / 2 - (i - (len(col) - 1) / 2) * 1.0
            pos[id(v)] = (d * 2.6, y)

    fig, ax = plt.subplots(figsize=(1.9 * ncols + 1.5, 1.1 * nrows + 1))
    ax.axis("off")
    ax.grid(False)
    bw, bh = 1.5, 0.62
    for v in nodes:
        x, y = pos[id(v)]
        # an op circle just left of the box for non-leaf nodes
        if v._prev:
            ax.add_patch(
                plt.Circle(
                    (x - bw / 2 - 0.32, y),
                    0.2,
                    facecolor=PALETTE["gold"],
                    edgecolor=PALETTE["ink"],
                    lw=0.8,
                )
            )
            ax.text(x - bw / 2 - 0.32, y, v._op, ha="center", va="center", fontsize=7)
            for c in v._prev:
                cx, cy = pos[id(c)]
                ax.add_patch(
                    FancyArrowPatch(
                        (cx + bw / 2, cy),
                        (x - bw / 2 - 0.52, y),
                        arrowstyle="-|>",
                        mutation_scale=8,
                        lw=0.8,
                        color=PALETTE["grey"],
                    )
                )
        box = FancyBboxPatch(
            (x - bw / 2, y - bh / 2),
            bw,
            bh,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white",
            edgecolor=PALETTE["green"] if not v._prev else PALETTE["ink"],
            lw=1.0,
        )
        ax.add_patch(box)
        name = v.label or v._op or "value"
        line1 = f"{name}"
        line2 = f"data {v.data:.4g}"
        line3 = f"grad {v.grad:.4g}" if show_grad else ""
        ax.text(x, y + 0.17, line1, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(x, y - 0.02, line2, ha="center", va="center", fontsize=6.5)
        if show_grad:
            ax.text(
                x, y - 0.20, line3, ha="center", va="center", fontsize=6.5, color=PALETTE["red"]
            )
    ax.set_xlim(-1.2, (ncols - 1) * 2.6 + 1.2)
    ax.set_ylim(-0.8, nrows - 0.2)
    if title:
        ax.set_title(title)
    return fig
