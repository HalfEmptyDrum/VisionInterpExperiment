# Running MechInterpVision on a CUDA GPU

The code is device-agnostic: `pick_device()` auto-selects `cuda` when available, and
attention uses a fused `scaled_dot_product_attention` fast-path (flash-attention on
CUDA) during training/eval — so a GPU is dramatically faster *and* the patch-4
memory blow-up we hit on MPS disappears. The interpretability path (cached attention
patterns, ablation, patching) is unchanged.

You do **not** need an A100 — these models are 0.1–3M params. A free Colab **T4**,
or a cheap **L4 / A10** (~$0.5–1/hr on Lambda/RunPod/Vast), is plenty.

## 1. Get the code onto the box

From this machine (don't copy `.venv` — rebuild it on the GPU):

```bash
cd /Users/kaifollmann/Projekte
zip -r mechvis.zip MechInterpVision -x 'MechInterpVision/.venv/*' 'MechInterpVision/figures/*' 'MechInterpVision/runs/*'
# scp mechvis.zip you@gpu-box:~/   (or push the git repo to a remote and clone)
```

On the GPU box:

```bash
unzip mechvis.zip && cd MechInterpVision
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # installs CUDA torch on a CUDA box
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Sanity check (≈10 s)

```bash
python -m mechvis.sanity                  # data montage + intensity baseline (≈ chance) + cache shapes
```

## 3. The full run → `report.html`

One command runs: grokking phase-diagram (first, guaranteed) → architecture sweep
(incl. the full patch-4 grid) → multi-seed of the top-2 → full Q1/Q2/Q3 mechanistic
analysis on the winner → a self-contained `report.html`.

```bash
python -u -m mechvis.overnight \
  --budget_hours 3 \
  --n_train 50000 --epochs 60 \
  --sweep_patches 8 4 --sweep_depths 2 3 4 --sweep_widths 64 128 \
  --grok_wds 0.01 0.1 1.0 --grok_sizes 500 1000 \
  --grok_steps 20000 --grok_depth 3 --grok_width 128
```

This is a **grok phase diagram** = `grok_wds × grok_sizes` cells (here 3×2 = 6) at
20k steps each on the interpretable d3/w128 model, plus a `sweep_patches × widths ×
depths` architecture grid (here 12 configs). Time estimates auto-scale for CUDA, and
a hard `--budget_hours` deadline (with `--reserve_min` held back for the report)
means it always finishes and writes the report; results are checkpointed
incrementally to `figures/*.json` and the run is resumable.

### Go bigger (if the GPU is fast / you have time)

```bash
  --grok_wds 0.0 0.01 0.03 0.1 0.3 1.0 3.0 --grok_sizes 300 600 1200 --grok_steps 50000
  --sweep_widths 64 128 256
```

### Knobs

| flag | meaning |
|---|---|
| `--budget_hours`, `--reserve_min` | hard wall-clock cap; report always written |
| `--sweep_patches/depths/widths` | architecture grid (cross-product) |
| `--grok_wds/sizes` | grokking phase-diagram axes (weight decay × train size) |
| `--grok_steps`, `--grok_depth`, `--grok_width` | grok run length + model size |
| `--n_train`, `--epochs` | sweep training budget per config |

## 4. Outputs

- `report.html` — open in a browser (figures embedded, fully self-contained).
- `figures/sweep_scatter.png`, `sweep_heatmap.png`, `grok_curves.png`, plus the
  analysis figures (`probe_trajectory`, `patch_filters`, `attn_examples`).
- `figures/overnight_results.json`, `grok_results.json` — raw numbers.
- `runs/sweep/*.pt` — checkpoints (re-analyze with `python -m mechvis.analysis --ckpt <path>`).

## Notes
- Bump batch size for throughput by editing `bs_for()` in `mechvis/overnight.py`
  (patch-8 → 256, patch-4 → 128 are MPS-safe defaults; on a big GPU you can raise both).
- Grokking is hyperparameter-sensitive — if a cell shows train→~1.0 with val flat then
  a delayed val rise, that's the grok. Widen `--grok_wds`/`--grok_sizes` around it.
