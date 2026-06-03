"""Architecture grid sweep: find configs that trade off accuracy vs interpretability.

    python -m mechvis.sweep --epochs 25

Grid (default): patch_size {8,4} x depth {2,3,4} x d_model {64,128}, n_heads=4.
All configs share one dataset and an equal training budget, so rankings are
comparable. For each config we record:

  acc, acc1           exact / within-1 val accuracy
  detect_auc          Q1: blob-presence AUC from a *patch_embed* token (detection
                      already in the embedding => cleaner mechanism)
  route_sel           Q2: CLS attention mass on blob patches above base rate,
                      best head per layer, averaged (clean object routing)
  acc_abstract        Q3: best layerwise CLS count-probe accuracy (count linearly
                      represented)
  transition_frac     Q3: depth fraction at which the count probe saturates
  n_tokens, params    complexity (interpretability cost)

We min-max normalise the three clarity proxies into `mechanism_clarity`, then
report the accuracy<->clarity Pareto frontier and a combined recommendation.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .analysis import attention_routing, collect, probe_auc, probe_clf
from .data import BlobConfig, centers_to_patch_counts, make_dataset
from .model import HookedViT, ViTConfig
from .train import fit, pick_device

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "figures"


def interp_metrics(model, ds, device, n_sub=1000):
    depth, d, P = model.cfg.depth, model.cfg.d_model, model.cfg.n_patches
    imgs = ds["images"][:n_sub]
    counts = ds["counts"][:n_sub]
    # per-patch labels at the model's grid (data grid may differ when sweeping patch_size)
    pcounts = np.stack([centers_to_patch_counts(ds["centers"][i], model.cfg.img_size,
                                                model.cfg.patch_size) for i in range(n_sub)])
    cls_keys = ["embed"] + [f"blocks.{i}.resid_post" for i in range(depth)] + ["resid_final"]
    acts = collect(model, imgs, device, list(dict.fromkeys(cls_keys + ["patch_embed"])))

    # Q3: per-layer CLS count probe -> max accuracy + saturation depth
    cls_acc = [probe_clf(acts[k][:, 0], counts) for k in cls_keys]
    abstract = max(cls_acc)
    thr = 0.95 * abstract
    ti = next((i for i, a in enumerate(cls_acc) if a >= thr), len(cls_acc) - 1)
    transition_frac = ti / (len(cls_keys) - 1)

    # Q1: blob presence decodable from a patch_embed token (AUC)
    presence = (pcounts > 0).reshape(-1)
    nrow = presence.shape[0]
    sel = np.random.default_rng(0).choice(nrow, min(nrow, 30000), replace=False)
    Xp = acts["patch_embed"][:, -P:, :].reshape(nrow, d).numpy()[sel]
    detect_auc = probe_auc(Xp, presence[sel])

    # Q2: attention object-selectivity (best head per layer, above base rate)
    rows = attention_routing(model, ds, device, n_sub=min(384, n_sub))
    base = rows[0]["blob_base_rate"]
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r["cls_mass_on_blobs"])
    route_sel = float(np.mean([max(v) for v in by_layer.values()]) - base)

    return {"acc_abstract": abstract, "transition_frac": transition_frac,
            "detect_auc": detect_auc, "route_sel": route_sel,
            "cls_acc_by_layer": [round(a, 3) for a in cls_acc]}


def _minmax(xs):
    a = np.asarray(xs, dtype=float)
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


def _pareto(acc, clar):
    """Indices on the accuracy<->clarity Pareto frontier (maximise both)."""
    pts = list(zip(acc, clar))
    keep = []
    for i, (ai, ci) in enumerate(pts):
        if not any(aj >= ai and cj >= ci and (aj > ai or cj > ci) for j, (aj, cj) in enumerate(pts) if j != i):
            keep.append(i)
    return keep


def plot_results(rows):
    acc = [r["acc"] for r in rows]
    clar = [r["mechanism_clarity"] for r in rows]
    front = set(_pareto(acc, clar))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {8: "tab:blue", 4: "tab:red"}
    for i, r in enumerate(rows):
        on = i in front
        ax.scatter(r["acc"], r["mechanism_clarity"], s=70 + r["n_tokens"] / 2,
                   c=colors[r["patch_size"]], edgecolors="k" if on else "none",
                   linewidths=1.6 if on else 0, alpha=0.85, zorder=3 if on else 2)
        ax.annotate(f"p{r['patch_size']}d{r['depth']}w{r['d_model']}",
                    (r["acc"], r["mechanism_clarity"]), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")
    fr = sorted([rows[i] for i in front], key=lambda r: r["acc"])
    ax.plot([r["acc"] for r in fr], [r["mechanism_clarity"] for r in fr], "k--", lw=1, alpha=0.5, zorder=1)
    ax.set_xlabel("exact val accuracy")
    ax.set_ylabel("mechanism clarity (mean of normalised detect/route/abstract)")
    ax.set_title("accuracy vs interpretability (point size ~ tokens; blue=patch8, red=patch4)")
    fig.tight_layout()
    p1 = f"{FIG}/sweep_scatter.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # accuracy heatmaps: depth x patch, per d_model
    dms = sorted({r["d_model"] for r in rows})
    depths = sorted({r["depth"] for r in rows})
    patches = sorted({r["patch_size"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(dms), figsize=(4 * len(dms), 3.2))
    for ax, dm in zip(np.atleast_1d(axes), dms):
        M = np.full((len(depths), len(patches)), np.nan)
        for r in rows:
            if r["d_model"] == dm:
                M[depths.index(r["depth"]), patches.index(r["patch_size"])] = r["acc"]
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=min(acc), vmax=max(acc))
        ax.set_xticks(range(len(patches)), [f"patch{p}" for p in patches])
        ax.set_yticks(range(len(depths)), [f"depth{d}" for d in depths])
        ax.set_title(f"d_model={dm}")
        for i in range(len(depths)):
            for j in range(len(patches)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="w", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("exact val accuracy")
    fig.tight_layout()
    p2 = f"{FIG}/sweep_heatmap.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    return p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--patches", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--d_models", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    torch.manual_seed(0)

    bcfg = BlobConfig()
    print(f"device={device}; generating shared data ...")
    train = make_dataset(bcfg, args.n_train, seed=1)
    val = make_dataset(bcfg, args.n_val, seed=2)
    analysis = make_dataset(bcfg, 1500, seed=123)

    grid = [(p, d, w) for p in args.patches for d in args.depths for w in args.d_models]
    print(f"{len(grid)} configs, {args.epochs} epochs each\n")
    rows = []
    for n, (patch, depth, d_model) in enumerate(grid):
        mcfg = ViTConfig(img_size=bcfg.img_size, patch_size=patch, in_ch=1, d_model=d_model,
                         depth=depth, n_heads=4, n_classes=bcfg.n_classes)
        model = HookedViT(mcfg).to(device)
        params = sum(p.numel() for p in model.parameters())
        best = fit(model, train, val, device, epochs=args.epochs, verbose=False)
        im = interp_metrics(model, analysis, device)
        row = {"patch_size": patch, "depth": depth, "d_model": d_model,
               "n_tokens": mcfg.n_patches + 1, "params": params,
               "acc": best["acc"], "acc1": best["acc1"], **im}
        rows.append(row)
        print(f"[{n+1:2d}/{len(grid)}] p{patch} d{depth} w{d_model:3d} "
              f"tok={row['n_tokens']:3d} | acc={row['acc']:.3f} acc1={row['acc1']:.3f} "
              f"| detect={im['detect_auc']:.3f} route={im['route_sel']:.3f} "
              f"abstract={im['acc_abstract']:.3f} trans={im['transition_frac']:.2f}")

    # composite scoring
    clif_d, clif_r, clif_a = _minmax([r["detect_auc"] for r in rows]), _minmax([r["route_sel"] for r in rows]), _minmax([r["acc_abstract"] for r in rows])
    for i, r in enumerate(rows):
        r["mechanism_clarity"] = float(np.mean([clif_d[i], clif_r[i], clif_a[i]]))
    acc_n = _minmax([r["acc"] for r in rows])
    for i, r in enumerate(rows):
        r["combined"] = float(0.6 * acc_n[i] + 0.4 * r["mechanism_clarity"])

    front = set(_pareto([r["acc"] for r in rows], [r["mechanism_clarity"] for r in rows]))
    for i in front:
        rows[i]["pareto"] = True

    p1, p2 = plot_results(rows)
    os.makedirs(FIG, exist_ok=True)
    with open(f"{FIG}/sweep_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)

    print("\n=== ranked by combined (0.6*acc + 0.4*clarity) ===")
    print(f"{'config':14s} {'acc':>6} {'acc1':>6} {'clarity':>8} {'comb':>6} {'tok':>4} {'pareto':>6}")
    for r in sorted(rows, key=lambda r: -r["combined"]):
        tag = f"p{r['patch_size']}d{r['depth']}w{r['d_model']}"
        print(f"{tag:14s} {r['acc']:.3f} {r['acc1']:.3f} {r['mechanism_clarity']:8.3f} "
              f"{r['combined']:.3f} {r['n_tokens']:4d} {'  *' if r.get('pareto') else '':>6}")
    best_acc = max(rows, key=lambda r: r["acc"])
    best_comb = max(rows, key=lambda r: r["combined"])
    print(f"\nhighest accuracy : p{best_acc['patch_size']}d{best_acc['depth']}w{best_acc['d_model']} ({best_acc['acc']:.3f})")
    print(f"best combined    : p{best_comb['patch_size']}d{best_comb['depth']}w{best_comb['d_model']} (acc {best_comb['acc']:.3f}, clarity {best_comb['mechanism_clarity']:.3f})")
    print(f"figures: {p1}, {p2}; results: {FIG}/sweep_results.json")


if __name__ == "__main__":
    main()
