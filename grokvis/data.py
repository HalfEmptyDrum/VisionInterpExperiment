"""Finite *target-detection counting* images for grokking a visual circuit.

Plain blob-counting cannot grok: count is a linear function of pixel intensity /
per-patch detections, so the network finds it instantly and generalizes without
ever memorizing (verified empirically). For grokking we need the pixel->label
map to be **non-linear**, and -- crucially -- we want that non-linearity to live
in the *vision encoder*, so the mechanistic story is about how the ViT *sees*,
not about post-hoc arithmetic.

Construction
------------
A ``grid x grid`` array of cells over a ``grid*cell_px`` image. **Every** cell is
occupied by exactly two dots placed in a 2x2 slot layout (slots TL, TR, BL, BR).
A cell is one of two kinds:

* **target**     -- the two dots lie on a *diagonal*: ``{TL,BR}`` or ``{TR,BL}``.
* **distractor** -- the two dots lie on a *side*:    ``{TL,TR}``, ``{BL,BR}``,
                    ``{TL,BL}`` or ``{TR,BR}``.

The **label is the number of target cells**. Both kinds use exactly two dots, so
*total intensity is identical across every image* -- the intensity / brightness
shortcut is dead by construction. More importantly, "diagonal vs side" is **not
linearly separable** in the four slot activations (it is an XOR/AND conjunction):
a single linear conv channel ``w1*TL+w2*TR+w3*BL+w4*BR`` cannot separate the two
diagonals from the four sides (the required inequalities are contradictory). So
the per-patch target detector *must* be built through the first block's GELU MLP
-- it cannot be read off the linear patch embedding. That non-linear, hard-to-
find-but-compressible detector is exactly the kind of circuit that memorizes
first and groks later.

Input space = ``6**(grid**2)`` distinct images (6 cell configs per cell): finite
and memorizable. Train and val are disjoint *images* sampled from this space, so
generalization to the held-out remainder requires learning the actual detector,
not the training images.

Layout matches ``mechvis``: ``(x, y) = (col, row)``; with ``patch_size ==
cell_px`` each patch token covers exactly one cell (row-major; sequence position
``t + 1`` after CLS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Slot indices within a cell: 0=TL, 1=TR, 2=BL, 3=BR  (x=col grows right, y=row grows down)
TARGET_PAIRS = [(0, 3), (1, 2)]                      # diagonals
DISTRACTOR_PAIRS = [(0, 1), (2, 3), (0, 2), (1, 3)]  # top, bottom, left, right
# Cell state code: 0,1 -> target variants;  2,3,4,5 -> distractor variants
PAIRS = TARGET_PAIRS + DISTRACTOR_PAIRS
N_STATES = len(PAIRS)            # 6 configs per cell
N_TARGET_STATES = len(TARGET_PAIRS)


@dataclass
class GridConfig:
    grid: int = 3              # cells per side -> grid**2 cells, label in 0..grid**2
    cell_px: int = 8           # pixels per cell -> img_size = grid * cell_px
    dot_sigma: float = 1.1     # std (px) of each dot
    dot_amp: float = 1.0       # peak amplitude of each dot
    slot_inset: float = 0.25   # slot centres at inset / (1-inset) of the cell span
    p_target: float = 0.5      # per-cell prob of being a target (vs distractor)
    jitter: float = 0.0        # optional sub-pixel jitter of dot centres (px); 0 => deterministic
    bg_noise: float = 0.0      # gaussian background noise std; 0 => deterministic input space
    max_total: int = 20000     # number of distinct images sampled from the 6**(grid**2) space
    train_frac: float = 0.5    # fraction of sampled images used for training
    seed: int = 0

    @property
    def img_size(self) -> int:
        return self.grid * self.cell_px

    @property
    def n_cells(self) -> int:
        return self.grid * self.grid

    @property
    def n_classes(self) -> int:
        return self.n_cells + 1  # target count 0..n_cells inclusive


def _sample_states(cfg: GridConfig, rng: np.random.Generator) -> np.ndarray:
    """Return an (N, n_cells) uint8 matrix of cell-state codes in 0..5.

    Enumerates the full ``6**n_cells`` space when it fits within ``max_total``;
    otherwise samples ``max_total`` distinct state vectors. Each cell is drawn as
    target (prob ``p_target``) vs distractor, then a uniform variant within kind,
    so the per-cell marginal over the 6 codes is controlled by ``p_target``.
    """
    n = cfg.n_cells
    full = N_STATES ** n
    if full <= cfg.max_total:
        # enumerate all state vectors via mixed-radix (base-6) counting
        idx = np.arange(full, dtype=np.int64)
        states = np.zeros((full, n), dtype=np.uint8)
        for c in range(n):
            states[:, c] = (idx // (N_STATES ** c)) % N_STATES
        return states

    target_codes = np.arange(0, N_TARGET_STATES)               # [0,1]
    distractor_codes = np.arange(N_TARGET_STATES, N_STATES)    # [2,3,4,5]
    seen: set[bytes] = set()
    rows: list[np.ndarray] = []
    # oversample in chunks and dedupe until we have max_total distinct vectors
    while len(rows) < cfg.max_total:
        m = cfg.max_total - len(rows)
        is_t = rng.random((m, n)) < cfg.p_target
        tv = target_codes[rng.integers(0, N_TARGET_STATES, size=(m, n))]
        dv = distractor_codes[rng.integers(0, len(distractor_codes), size=(m, n))]
        chunk = np.where(is_t, tv, dv).astype(np.uint8)
        for row in chunk:
            b = row.tobytes()
            if b not in seen:
                seen.add(b)
                rows.append(row)
                if len(rows) >= cfg.max_total:
                    break
    return np.asarray(rows, dtype=np.uint8)


def _slot_centres(cfg: GridConfig, row: int, col: int) -> np.ndarray:
    """Pixel centres of the 4 slots (TL,TR,BL,BR) for cell (row, col)."""
    cp, i = cfg.cell_px, cfg.slot_inset
    x0, y0 = col * cp, row * cp
    lo_x, hi_x = x0 + i * cp, x0 + (1 - i) * cp
    lo_y, hi_y = y0 + i * cp, y0 + (1 - i) * cp
    return np.array([[lo_x, lo_y], [hi_x, lo_y], [lo_x, hi_y], [hi_x, hi_y]], dtype=np.float32)


def _render(states: np.ndarray, cfg: GridConfig, rng: np.random.Generator) -> np.ndarray:
    """Render one cell-state vector (length n_cells) to an (img, img) float image."""
    H = cfg.img_size
    ys, xs = np.mgrid[0:H, 0:H].astype(np.float32)  # ys=row, xs=col
    img = np.zeros((H, H), dtype=np.float32)
    s = cfg.dot_sigma
    for t, code in enumerate(states):
        row, col = t // cfg.grid, t % cfg.grid
        centres = _slot_centres(cfg, row, col)
        for slot in PAIRS[int(code)]:
            cx, cy = centres[slot]
            if cfg.jitter > 0:
                cx += rng.uniform(-cfg.jitter, cfg.jitter)
                cy += rng.uniform(-cfg.jitter, cfg.jitter)
            g = cfg.dot_amp * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * s * s)))
            img = np.maximum(img, g)
    if cfg.bg_noise > 0:
        img = img + rng.normal(0.0, cfg.bg_noise, img.shape).astype(np.float32)
    np.clip(img, 0.0, 1.0, out=img)
    return img


def make_grid_dataset(cfg: GridConfig, seed: Optional[int] = None) -> dict:
    """Build the finite target-detection dataset and a fixed train/val split.

    Returns a dict of tensors plus metadata:
      images        (N, 1, H, W) float32 in [0, 1]
      labels        (N,)         int64   number of TARGET cells (the training target)
      counts        (N,)         int64   alias of labels (compat with mechvis analysis)
      is_target     (N, n_cells) uint8   per-cell target indicator (Q1 probe target)
      patch_counts  (N, n_cells) int64   alias of is_target (compat with mechvis probes)
      states        (N, n_cells) uint8   per-cell config code 0..5 (full input description)
      train_idx     (n_tr,)      int64   train-split indices into images
      val_idx       (n_va,)      int64   held-out remainder
      cfg                                the GridConfig used
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    states = _sample_states(cfg, rng)
    n, H = states.shape[0], cfg.img_size
    imgs = np.zeros((n, H, H), dtype=np.float32)
    for i in range(n):
        imgs[i] = _render(states[i], cfg, rng)
    is_target = (states < N_TARGET_STATES).astype(np.uint8)   # codes 0,1 are targets
    labels = is_target.sum(axis=1).astype(np.int64)

    perm = rng.permutation(n)
    n_tr = int(round(cfg.train_frac * n))
    train_idx = np.sort(perm[:n_tr])
    val_idx = np.sort(perm[n_tr:])

    labels_t = torch.from_numpy(labels)
    is_target_t = torch.from_numpy(is_target)
    return {
        "images": torch.from_numpy(imgs).unsqueeze(1),
        "labels": labels_t,
        "counts": labels_t,                                   # alias
        "is_target": is_target_t,
        "patch_counts": is_target_t.long(),                   # alias for mechvis probes
        "states": torch.from_numpy(states),
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "cfg": cfg,
    }


def split_tensors(ds: dict) -> tuple[dict, dict]:
    """Materialise ``ds`` into ``(train, val)`` dicts (images/labels/is_target/states)."""
    keys = ("images", "labels", "is_target", "states")
    tr, va = ds["train_idx"], ds["val_idx"]
    train = {k: ds[k][tr] for k in keys}
    val = {k: ds[k][va] for k in keys}
    return train, val


def intensity_baseline(images: torch.Tensor, labels: torch.Tensor) -> dict:
    """How well does summed pixel intensity alone predict the target count?

    By construction every image has the same number of dots, so total intensity
    is ~constant and carries no information about the label -- this should sit at
    chance (the linear/brightness shortcut is dead). Reported for transparency.
    """
    x = images.reshape(images.shape[0], -1).sum(1).double().numpy()
    y = labels.double().numpy()
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum()) + 1e-12
    r = float((xc * yc).sum() / denom)
    a = (xc * yc).sum() / ((xc**2).sum() + 1e-12)
    b = y.mean() - a * x.mean()
    pred = np.clip(np.rint(a * x + b), labels.min().item(), labels.max().item())
    acc = float((pred == y).mean())
    # also report the majority-class rate, the real "chance" level for this task
    vals, cnts = np.unique(y, return_counts=True)
    return {"pearson_r": r, "r2": r**2, "lin_round_acc": acc,
            "majority_rate": float(cnts.max() / len(y))}
