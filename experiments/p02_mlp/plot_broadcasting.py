"""Draw broadcasting: how a (N, in) @ (in, out) product and a (out,) bias line up."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dl.viz import PALETTE, save_figure, use_style

use_style()
FIG = Path("docs/02-pytorch-mlp/figures")


def grid(ax, x, y, rows, cols, color, label, cell=0.32, alpha=1.0):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x + c * cell, y - (r + 1) * cell),
                    cell,
                    cell,
                    facecolor=color,
                    edgecolor="white",
                    lw=0.8,
                    alpha=alpha,
                )
            )
    ax.text(x + cols * cell / 2, y + 0.12, label, ha="center", va="bottom", fontsize=8)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax in axes:
    ax.axis("off")
    ax.grid(False)

# Left: one layer as a matrix product
ax = axes[0]
grid(ax, 0.0, 2.0, 4, 3, PALETTE["blue"], "x  (N=4, in=3)")
ax.text(1.15, 1.35, "@", fontsize=14, ha="center", va="center")
grid(ax, 1.35, 2.0, 3, 5, PALETTE["green"], "Wᵀ  (in=3, out=5)")
ax.text(3.15, 1.35, "+", fontsize=14, ha="center", va="center")
grid(ax, 3.35, 2.0, 1, 5, PALETTE["gold"], "b  (out=5)")
grid(ax, 3.35, 1.68, 3, 5, PALETTE["gold"], "", alpha=0.25)
ax.annotate(
    "",
    xy=(4.3, 0.55),
    xytext=(4.3, 1.55),
    arrowprops={"arrowstyle": "->", "color": PALETTE["grey"]},
)
ax.text(4.55, 1.05, "copied\ndown N rows", fontsize=7, va="center", color=PALETTE["grey"])
ax.text(5.35, 1.35, "=", fontsize=14, ha="center", va="center")
grid(ax, 5.6, 2.0, 4, 5, PALETTE["red"], "y  (N=4, out=5)")
ax.set_xlim(-0.2, 7.4)
ax.set_ylim(0.3, 2.5)
ax.set_title("one Linear layer on a batch: x @ Wᵀ + b", fontsize=9)

# Right: the rule
ax = axes[1]
rules = [
    ("(4, 3) + (3,)", "→ (4, 3)", "trailing dims match; the (3,) row is copied 4 times"),
    ("(4, 1) + (1, 3)", "→ (4, 3)", "size-1 dims stretch to match the other side"),
    ("(4, 3) + (4,)", "error", "3 vs 4 on the last dim: neither is 1"),
    ("(2, 1, 5) * (3, 1)", "→ (2, 3, 5)", "align from the right, pad missing dims with 1"),
]
ax.text(0, 1.0, "the rule, aligned from the right:", fontsize=9, va="top", fontweight="bold")
for i, (lhs, rhs, why) in enumerate(rules):
    y = 0.78 - i * 0.2
    ax.text(0.02, y, lhs, fontsize=8.5, family="monospace", va="center")
    ax.text(
        0.42,
        y,
        rhs,
        fontsize=8.5,
        family="monospace",
        va="center",
        color=PALETTE["red"] if rhs == "error" else PALETTE["green"],
    )
    ax.text(0.02, y - 0.08, why, fontsize=7, va="center", color=PALETTE["grey"])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_title("broadcasting rules", fontsize=9)
fig.tight_layout()
print(save_figure(fig, FIG / "broadcasting.png"))
