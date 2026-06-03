# MechInterpVision

Mechanistic interpretability of **object counting** in a tiny vision transformer.
We train a small patch-based ViT from scratch to count Gaussian blobs in synthetic
images, then reverse-engineer how it turns raw pixels into a count:

1. **How patch embeddings detect blobs** (Q1)
2. **How attention heads route spatial information** (Q2)
3. **Where the spatial→abstract transition happens** (Q3)

## Task design

Synthetic 64×64 grayscale images with **free-floating** 2-D Gaussian blobs
(continuous pixel values). Label = blob count, framed as **classification over
0..8**. Readout from a dedicated **CLS token**.

Key design choices and why:

- **Free-floating blobs** (continuous centres, min-separation rejection sampling):
  blobs routinely straddle 8×8 patch boundaries, so a single patch embedding
  cannot see a whole blob — cross-patch integration via attention is *required*,
  making Q2 non-trivial.
- **Constant-total-intensity normalization** (`normalize="const_total"`): each
  image with ≥1 blob is rescaled to a fixed total intensity. This **decorrelates
  global brightness from count** so the model cannot solve the task by integrating
  pixel intensity — it must detect and count discrete objects. We verify this with
  `intensity_baseline()`: a linear predictor from summed intensity sits at chance
  (R²≈0.1, round-acc≈0.11 ≈ 1/9). Any residual per-blob brightness cue must route
  through detection, which is exactly the mechanism we study.
- **Wide, independent amplitude/σ jitter**: per-blob mass varies, removing
  size/brightness regularities that could leak count.

## Model

`HookedViT` (see [mechvis/model.py](mechvis/model.py)): pre-LN, learned positional
embeddings, Conv2d patch projection, CLS-token classification head. Defaults:
`d_model=128, depth=3, n_heads=4` (~0.6M params) — small enough to fully
reverse-engineer.

Instrumentation:
- `run_with_cache(x) -> (logits, cache)` — every intermediate activation
  (patch_embed, per-block resid_pre/mid/post, attention patterns, per-head `z`,
  MLP activations, CLS, logits).
- `run_with_hooks(x, hooks={name: fn}, ablate_pos=...)` — one mechanism for both
  **ablation** (zero a head's `z`) and **activation patching** (overwrite residual
  positions from a source run).
- `head_results(cache, layer)` — per-head contribution to the residual stream.

## Analyses → questions

See [mechvis/analysis.py](mechvis/analysis.py):

| Question | Analysis |
|---|---|
| Q1 patch embeddings detect blobs | conv-filter viz; per-patch-token probe for local blob presence (AUC) / count (R²) — high at `patch_embed` ⇒ detection lives in the embedding |
| Q2 attention routes spatial info | CLS→patch attention maps; CLS attention mass on blob patches vs base rate; within-blob uniformity (CV); per-head ablation; positional-embedding ablation |
| Q3 spatial→abstract transition | layerwise CLS count probe + logit-lens (when count becomes decodable); patch-token info decay; activation patching (which layer/site carries the movable count) |

## Setup & run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m mechvis.sanity                 # data montage, baseline, shape/cache check
.venv/bin/python -m mechvis.train --epochs 40 --n_train 40000
.venv/bin/python -m mechvis.analysis --ckpt runs/vit_count.pt
```

Figures and a `metrics.json` are written to `figures/`.

## Status

Scaffolding complete (data, model, training, full analysis suite, sanity-tested
end-to-end). Training the final model and running the analysis pass next; results
to be summarized here.
