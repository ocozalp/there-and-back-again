# There and Back Again: deep learning, one chapter at a time

This repository is a personal, documentation-first walk through deep learning. It starts with a
scalar autograd engine written by hand and ends with a small text-to-image model that illustrates a
diary. Every model is tiny (under about 5M parameters), trains in minutes on a laptop, and exists
to make one chapter of the documentation concrete and testable.

**The documentation is the product.** The code is here so the chapters can show real numbers,
real figures, and real failures. Start reading at [docs/00-conventions.md](docs/00-conventions.md).

## The scenario

Bilbo, a hobbit, keeps a diary during his journey across Middle-earth and back. The project
generates its own data from that story:

- **Images.** A renderer draws Bilbo as a fixed composition of simple shapes (round face, brown
  curly hair, green hooded cloak, red waistcoat, bare feet, walking stick, pipe) inside scenes
  whose attributes vary: location, weather, time of day, companions, and objects. Every attribute
  is a parameter, so the ground truth is always known: Bilbo's look must stay fixed, the scene may
  change.
- **Text.** Each scene comes with a diary entry in Bilbo's voice, produced by a template generator
  from the same scene parameters. Tolkien's books are under copyright and are not used as training
  data. The language-model phases pretrain on the public-domain
  [Diary of Samuel Pepys](https://www.gutenberg.org/ebooks/4200) and meet Bilbo's entries later.
- **The twist.** Midway through the journey Bilbo meets Gollum, who looks nothing like anyone the
  model has seen, and only a dozen pictures of him exist. Teaching the model Gollum from a few
  images is the LoRA task.

The final deliverable is an illustrated diary: thirty consecutive entries written by a fine-tuned
language model, each illustrated by a fine-tuned image model, with Bilbo drawn consistently
throughout while the scenery changes from the Shire to the Lonely Mountain and home.

## Phases

| Phase | Chapter | What it teaches | Library equivalent | Status |
|---|---|---|---|---|
| 0 | [Skeleton and scenario](docs/00-skeleton/README.md) | Tooling, conventions, the renderer and caption generator | `uv`, `pytest`, `ruff` | done |
| 1 | [Autograd from scratch](docs/01-autograd/README.md) | Computational graphs, backprop, gradient checking | `torch.autograd` | done |
| 2 | [Tensors and PyTorch](docs/02-pytorch-mlp/README.md) | Tensors, `nn.Module`, optimizers, the train loop | `torch.nn`, `torch.optim` | done |
| 3 | Language modeling before attention | Tokens, embeddings, bigram and MLP language models | `nn.Embedding`, `tiktoken` | next |
| 4 | Attention and a small GPT | Self-attention, the transformer block, sampling | `transformers` GPT-2 | planned |
| 5 | Vision models and latent spaces | CNNs, ViTs, autoencoders, VAEs | `torchvision`, `timm`, `AutoencoderKL` | planned |
| 6 | Connecting vision and language | Contrastive learning, cross-attention, captioning | `CLIPModel`, BLIP | planned |
| 7 | Generative models on pixels | DDPM, DDIM, guidance, flow matching | `diffusers` U-Net and schedulers | planned |
| 8 | Latent text-to-image | The anatomy of Stable Diffusion, built from our parts | `StableDiffusionPipeline` | planned |
| 9 | LoRA and customization | Low-rank adapters, textual inversion, forgetting | `peft`, `load_lora_weights` | planned |
| 10 | Controlling randomness | Seeds, inpainting, spatial control, the illustrated diary | inpaint and ControlNet pipelines | planned |

## Layout

```
docs/          chapters, one folder per phase, figures next to the text
src/dl/        hand-written, reusable code (never imports transformers, diffusers, or peft)
  world/       the scenario: Bilbo, the attribute vocabulary, renderer, captions, dataset
  autograd/    the Phase 1 scalar engine, kept as a reference
  nn/ models/ training/ sampling/ viz/   grow phase by phase
experiments/   per-phase scripts: train_*.py, plot_*.py, results.md, and library.py
tests/         shape tests, numerical checks, renderer and caption agreement
```

## Running

```bash
uv sync --group dev            # core environment
uv sync --group dev --extra library   # add the production libraries for library.py scripts
uv run pytest
uv run ruff check
uv run python experiments/p00_skeleton/check_device.py
```

Budgets that every phase respects: under 5M parameters, under 15 minutes of training on the
Metal backend, downloaded datasets under 200 MB.
