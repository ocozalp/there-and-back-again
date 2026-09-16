# Conventions: how this project is written

This file is the contract every chapter follows. Read it once; after that the chapters should feel
predictable.

## Why the documentation comes first

This repository exists so that future-me can come back after months away and relearn deep learning
from the fundamentals, with working code beside every idea. Code that trains a model but leaves no
explanation behind has failed the project's purpose, even if the model works.

## The story

Every phase advances one scenario: Bilbo keeps a diary on his journey across Middle-earth, and by
the end the project writes and illustrates that diary with models it trained itself. The scenario
is defined in one place, [`src/dl/world/scenario.py`](../src/dl/world/scenario.py). Each chapter
opens by saying where it sits on that road.

## Chapter template

Every chapter lives at `docs/NN-name/README.md` with its figures in `docs/NN-name/figures/` and
has these nine sections, in this order, with these headings:

1. **The question this phase answers.** Two or three sentences, plus where it sits on the road to
   the illustrated diary.
2. **Fundamentals.** The theory, with math, readable without opening the code. Every concept gets
   a diagram. Written for someone comfortable with linear algebra and calculus who has not done
   deep learning.
3. **What we built.** A walkthrough of the code, linking to files and functions, with an
   architecture diagram that shows tensor shapes on the arrows.
4. **Experiments and results.** Parameter counts, train time, loss curves, samples, a results
   table. Numbers come from `experiments/pNN_*/results.md`, never from memory.
5. **What went wrong and what surprised me.** Bugs, dead ends, and intuitions that turned out to be
   wrong. This is often the most useful section on a re-read.
6. **The library equivalent.** Which production API does the same job (PyTorch, `transformers`,
   `diffusers`, `peft`), a table mapping our classes and arguments onto the library's, what the
   library adds that we skipped and why it exists, and the output of `library.py` next to ours.
7. **Exit criteria.** A checklist of measurable outcomes. The phase is finished when every box is
   checked, and not before.
8. **Further reading.** Linked references with one sentence each on why it matters.
9. **Glossary additions.** New terms defined here and appended to [glossary.md](glossary.md).

## Depth gradient

Early chapters are short because the ideas are small. Later chapters are long because the ideas
are layered and the failure modes matter.

| Phases | Target length | Minimum figures | Style |
|---|---|---|---|
| 0 to 3 | 1,500 to 3,000 words | 4 | Short. State the idea, show it working, move on. |
| 4 to 6 | 4,000 to 7,000 words | 8 | Full derivations, worked numeric examples, ablations. |
| 7 to 10 | 7,000 to 12,000 words | 12 | Exhaustive. Derivations, failure galleries, variant comparisons, decision guides. |

## Figures and diagrams

Chapters are meant to be read visually. Two kinds of figures, both committed under `figures/`:

- **Generated figures** (loss curves, sample grids, attention maps, embedding plots) come from a
  `plot_*.py` or `render_*.py` script in the phase's experiment folder, so they can always be
  regenerated. Shared plotting lives in [`src/dl/viz`](../src/dl/viz) so every chapter looks the same.
- **Drawn diagrams** (architecture and concept diagrams) are SVG or PNG. When drawn by hand in
  Excalidraw, the `.excalidraw` source sits next to the export. Simple flows may be Mermaid blocks
  inline in the markdown.

Every architecture gets a block diagram with tensor shapes on the arrows. Every generative phase
includes a sample grid per experiment and a failure gallery.

## Links

Every paper, video, blog post, library page, or dataset mentioned anywhere is a markdown link on
first mention. [reading-list.md](reading-list.md) collects all of them, grouped by phase, with one
sentence each.

## Code rules

- `src/dl` is hand-written, reusable code. It never imports `transformers`, `diffusers`, or `peft`.
  Those libraries appear only in `experiments/pNN_*/library.py` and in comparison tests, installed
  with `uv sync --group dev --extra library`.
- `experiments/pNN_*/` holds phase-specific scripts: `train_*.py`, `plot_*.py`, `library.py`, and a
  `results.md` that records config, parameter count, wall-clock time, final metrics, and the
  commit hash.
- **Hand-write once, then use the library.** A component is written by hand in the chapter that
  teaches it and compared with the library version there. Later chapters use the library version
  (`nn.Linear` from Phase 3 on, `F.cross_entropy`, `torch.optim.Adam`) unless the component is that
  chapter's own subject. The point is to understand each piece, not to maintain a parallel framework.
- Tests in `tests/` cover shapes for each new layer, numerical checks against a reference where
  one exists (finite differences, PyTorch, a naive loop), and agreement between the renderer and
  the caption generator.
- `uv run ruff check` and `uv run pytest` pass at the end of every phase.

## Budgets

| Resource | Limit |
|---|---|
| Parameters per model | about 5M |
| Training time per experiment | about 15 minutes on the Metal backend |
| Downloaded datasets | under 200 MB |
| Pretrained checkpoints for `library.py` | under 2 GB total, cached outside the repository |

A chapter explains any exception before the budget is raised.

## Git

One tag per completed phase (`phase-00`, `phase-01`, ...) so any chapter can be read against the
exact code that produced its numbers. Commit messages mention the phase.
