"""Draw the computation graph of one neuron, before and after backward()."""

from pathlib import Path

from dl.autograd import Value
from dl.autograd.draw import draw_graph
from dl.viz import save_figure

FIG = Path("docs/01-autograd/figures")

# One neuron with two inputs, as in the chapter's worked example
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")
x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
s = x1w1 + x2w2
s.label = "x1*w1 + x2*w2"
n = s + b
n.label = "n"
o = n.tanh()
o.label = "o"

save_figure(
    draw_graph(o, "forward pass: every value remembers how it was made", show_grad=False),
    FIG / "graph_forward.png",
)
o.backward()
save_figure(
    draw_graph(o, "after backward(): grad = d o / d node, filled from right to left"),
    FIG / "graph_backward.png",
)

# A value used twice: gradients accumulate
a = Value(3.0, label="a")
b2 = a * a
b2.label = "a*a"
c = b2 + a
c.label = "a*a + a"
c.backward()
save_figure(
    draw_graph(c, "a is used twice, so its grad is the sum of two paths: 2a + 1 = 7"),
    FIG / "graph_accumulate.png",
)
print("wrote graph figures")
