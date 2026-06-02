"""Plotting helpers (headless / Agg backend)."""
from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def montage(images, counts, path: str, n: int = 16, ncol: int = 4):
    """Save an n-image grid with count titles."""
    imgs = _to_np(images)
    if imgs.ndim == 4:  # (N, 1, H, W)
        imgs = imgs[:, 0]
    counts = _to_np(counts)
    n = min(n, imgs.shape[0])
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 2.0))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        if i < n:
            # per-image display scaling: const-total images are dim, so vmax=1 would
            # render them near-black. The model still sees true [0,1] values.
            vmax = max(float(imgs[i].max()), 1e-6)
            ax.imshow(imgs[i], cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"k={int(counts[i])}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
