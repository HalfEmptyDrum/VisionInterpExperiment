"""grokvis — a *finite*, memorizable vision-counting task for studying grokking.

The blob-counting task in ``mechvis`` draws every image from a continuous
distribution (jittered centres / sigma / amplitude), so the training set is
effectively infinite and never repeats: the model fits the distribution smoothly
and never enters the "memorize a finite set, then generalize" regime that
grokking requires.

``grokvis`` instead defines a **finite** input space whose pixel->label map is
**non-linear**, with the non-linearity living in the *vision encoder*: a
``G x G`` grid of cells, every cell holding two dots that form either a *diagonal*
(target) or a *side* (distractor) pair. The label is the number of target cells.
"diagonal vs side" is not linearly separable, so the per-patch detector must be
built through the first block's GELU MLP -- a non-linear visual circuit that the
model memorizes first (train acc -> 1.0 while val sits near chance) and "groks"
later (val acc jumps to ~1.0 long after train saturates). Input space is
``6^(G*G)`` distinct images (finite, memorizable). See ``data.py`` for details.

It reuses :class:`mechvis.model.HookedViT`, so every Q1/Q2/Q3 analysis in
``mechvis.analysis`` applies directly to a grokked checkpoint.
"""
from .data import GridConfig, make_grid_dataset

__all__ = ["GridConfig", "make_grid_dataset"]
