"""Synthetic Gaussian-blob images for object-counting interpretability.

Free-floating 2-D Gaussian blobs on a continuous-valued canvas. The label is the
number of blobs. Amplitude and sigma are jittered widely and *independently*, so
that total image intensity is a poor predictor of count -- this pushes the model
toward discrete detection-and-sum rather than intensity integration. Use
`intensity_baseline` to quantify exactly how much of the count is recoverable
from summed intensity alone (we want this to be low).

Blob centres are placed at continuous coordinates with a minimum separation, so
blobs routinely straddle patch boundaries -- making cross-patch integration
(the attention-routing question) non-trivial.

Conventions
-----------
* Coordinates are ``(x, y)`` with ``x`` the column and ``y`` the row.
* Patch tokens are row-major: token ``t`` (0-indexed over patches) covers
  ``row = t // grid``, ``col = t % grid``. This matches a ``Conv2d`` patch embed
  flattened with ``rearrange('b d gh gw -> b (gh gw) d')``. Note the model
  prepends a CLS token, so patch ``t`` lives at sequence position ``t + 1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class BlobConfig:
    img_size: int = 64
    patch_size: int = 8
    min_count: int = 0
    max_count: int = 8  # counts in [min_count, max_count] -> max_count+1 classes
    sigma_range: tuple[float, float] = (1.0, 2.5)
    amp_range: tuple[float, float] = (0.3, 1.0)
    min_dist: float = 9.0  # min centre-to-centre distance (px); keeps blobs separable
    margin: float = 5.0  # keep centres this far from the edge (>= ~1.5 sigma_max)
    bg_noise: float = 0.01  # gaussian background noise std
    combine: str = "max"  # "max" or "add" for overlapping blobs
    # "const_total" rescales each k>=1 image to a fixed total intensity, decorrelating
    # global brightness from count (the intensity-integration shortcut). "none" leaves it.
    normalize: str = "const_total"
    intensity_total: float = 5.0
    seed: int = 0

    @property
    def n_classes(self) -> int:
        return self.max_count + 1

    @property
    def grid(self) -> int:
        return self.img_size // self.patch_size

    @property
    def n_patches(self) -> int:
        return self.grid**2


def _sample_centers(rng: np.random.Generator, k: int, cfg: BlobConfig) -> np.ndarray:
    """Rejection-sample ``k`` centres with pairwise distance >= ``cfg.min_dist``."""
    lo, hi = cfg.margin, cfg.img_size - cfg.margin
    centers: list[np.ndarray] = []
    tries = 0
    max_tries = 2000
    while len(centers) < k and tries < max_tries:
        tries += 1
        c = rng.uniform(lo, hi, size=2)
        if all(np.hypot(*(c - p)) >= cfg.min_dist for p in centers):
            centers.append(c)
    if len(centers) < k:
        raise RuntimeError(
            f"could not place {k} blobs with min_dist={cfg.min_dist} in "
            f"{cfg.img_size}px (placed {len(centers)}). Lower min_dist/max_count."
        )
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def _render(
    centers: np.ndarray, sigmas: np.ndarray, amps: np.ndarray, cfg: BlobConfig
) -> np.ndarray:
    H = cfg.img_size
    ys, xs = np.mgrid[0:H, 0:H].astype(np.float32)  # ys=row, xs=col
    img = np.zeros((H, H), dtype=np.float32)
    for (cx, cy), s, a in zip(centers, sigmas, amps):
        g = a * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * s * s)))
        img = np.maximum(img, g) if cfg.combine == "max" else img + g
    return img


def _patch_counts(centers: np.ndarray, cfg: BlobConfig) -> np.ndarray:
    """Number of blob *centres* falling in each patch (row-major, length n_patches)."""
    pc = np.zeros(cfg.n_patches, dtype=np.int64)
    g, P = cfg.grid, cfg.patch_size
    for cx, cy in centers:
        col = min(int(cx // P), g - 1)
        row = min(int(cy // P), g - 1)
        pc[row * g + col] += 1
    return pc


def make_dataset(cfg: BlobConfig, n: int, seed: Optional[int] = None) -> dict:
    """Generate ``n`` samples.

    Returns a dict of tensors plus python-side metadata:
      images        (n, 1, H, W) float32 in [0, 1]
      counts        (n,)          int64   total blob count (the label)
      patch_counts  (n, P)        int64   per-patch centre counts (probe target)
      centers       list[np.ndarray]      (k_i, 2) centre coords per image
      sigmas/amps   list[np.ndarray]      per-blob params per image
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    imgs = np.zeros((n, cfg.img_size, cfg.img_size), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int64)
    patch_counts = np.zeros((n, cfg.n_patches), dtype=np.int64)
    centers_all, sigmas_all, amps_all = [], [], []

    for i in range(n):
        k = int(rng.integers(cfg.min_count, cfg.max_count + 1))
        if k == 0:
            centers = np.zeros((0, 2), dtype=np.float32)
            sigmas = np.zeros(0, dtype=np.float32)
            amps = np.zeros(0, dtype=np.float32)
            img = np.zeros((cfg.img_size, cfg.img_size), dtype=np.float32)
        else:
            centers = _sample_centers(rng, k, cfg)
            sigmas = rng.uniform(*cfg.sigma_range, size=k).astype(np.float32)
            amps = rng.uniform(*cfg.amp_range, size=k).astype(np.float32)
            img = _render(centers, sigmas, amps, cfg)
            if cfg.normalize == "const_total":
                s = float(img.sum())
                if s > 0:
                    img *= cfg.intensity_total / s
        if cfg.bg_noise > 0:
            img = img + rng.normal(0.0, cfg.bg_noise, img.shape).astype(np.float32)
        np.clip(img, 0.0, 1.0, out=img)

        imgs[i] = img
        counts[i] = k
        patch_counts[i] = _patch_counts(centers, cfg)
        centers_all.append(centers)
        sigmas_all.append(sigmas)
        amps_all.append(amps)

    return {
        "images": torch.from_numpy(imgs).unsqueeze(1),
        "counts": torch.from_numpy(counts),
        "patch_counts": torch.from_numpy(patch_counts),
        "centers": centers_all,
        "sigmas": sigmas_all,
        "amps": amps_all,
    }


def intensity_baseline(images: torch.Tensor, counts: torch.Tensor) -> dict:
    """How well does summed pixel intensity alone predict the count?

    Fits count ~ a * total_intensity + b (closed form) and reports Pearson r,
    R^2, and the accuracy of the rounded linear predictor. Low values mean the
    task is *not* trivially solvable by intensity integration.
    """
    x = images.reshape(images.shape[0], -1).sum(1).double().numpy()
    y = counts.double().numpy()
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum()) + 1e-12
    r = float((xc * yc).sum() / denom)
    a = (xc * yc).sum() / ((xc**2).sum() + 1e-12)
    b = y.mean() - a * x.mean()
    pred = np.clip(np.rint(a * x + b), counts.min().item(), counts.max().item())
    acc = float((pred == y).mean())
    return {"pearson_r": r, "r2": r**2, "lin_round_acc": acc}
