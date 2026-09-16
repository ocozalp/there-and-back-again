"""Pick the compute device.

On this project's machine that is Apple's Metal backend (``mps``). The CPU fallback keeps every
script runnable anywhere, only slower.
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return ``prefer`` if given, else mps, else cuda, else cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
