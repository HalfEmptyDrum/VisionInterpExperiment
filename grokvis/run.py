"""Run a grokking experiment on the finite target-detection counting task.

The label is the number of *target* cells (two dots on a diagonal) among
distractor cells (two dots on a side). "diagonal vs side" is not linearly
separable, so the per-patch detector must be built through the encoder's GELU
MLP -- a non-linear visual circuit that memorizes first and groks later.

    # quick demo (3x3 grid)
    python -m grokvis.run --grid 3 --steps 40000

    # bigger grid / longer run
    python -m grokvis.run --grid 4 --train_frac 0.4 --steps 80000 --wd 1.0

Writes a train-vs-val curve (figures/grokvis/<tag>_curve.png), a sample montage,
a history json, and a best-val checkpoint (runs/grokvis/<tag>.pt) ready for
mechvis analysis.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from mechvis.model import HookedViT, ViTConfig

from .data import GridConfig, intensity_baseline, make_grid_dataset, split_tensors
from .train import fit_grok, grok_summary, pick_device

FIG = "figures/grokvis"
RUNDIR = "runs/grokvis"


def tag_of(args) -> str:
    return (f"g{args.grid}_f{args.train_frac}_pt{args.p_target}_wd{args.wd}_lr{args.lr}"
            f"_d{args.depth}w{args.d_model}").replace(".", "p")


def plot_curve(hist, summary, tag, path):
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(steps, [h["train_acc"] for h in hist], color="tab:blue", label="train")
    ax1.plot(steps, [h["val_acc"] for h in hist], color="tab:red", label="val")
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("accuracy")
    ax2.plot(steps, [h["train_loss"] for h in hist], color="tab:blue", label="train")
    ax2.plot(steps, [h["val_loss"] for h in hist], color="tab:red", label="val")
    ax2.set_ylabel("cross-entropy loss")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("optimization step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    for ax, key, col in ((ax1, "memorize_step", "tab:blue"), (ax1, "generalize_step", "tab:red")):
        s = summary.get(key)
        if s:
            ax.axvline(s, color=col, ls="--", lw=1, alpha=0.6)
    gtxt = (f"grokked (gap={summary['grok_gap']})" if summary["grokked"]
            else f"no clean grok (mem@{summary['memorize_step']}, gen@{summary['generalize_step']})")
    fig.suptitle(f"{tag}   |   train_final={summary['train_final']:.3f}  "
                 f"val_best={summary['val_best']:.3f}  |  {gtxt}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def montage(ds, path, k=16):
    import matplotlib.pyplot as plt
    import numpy as np
    n = ds["images"].shape[0]
    sel = np.linspace(0, n - 1, min(k, n)).astype(int)
    cols = 8
    rows = (len(sel) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6), squeeze=False)
    for ax, i in zip(axes.flat, sel):
        ax.imshow(ds["images"][i, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"#tgt={int(ds['labels'][i])}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(sel):]:
        ax.axis("off")
    fig.suptitle("target-detection samples (diagonal=target, side=distractor)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--cell_px", type=int, default=8)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--max_total", type=int, default=20000)
    ap.add_argument("--p_target", type=float, default=0.5)
    ap.add_argument("--dot_sigma", type=float, default=1.1)
    ap.add_argument("--slot_inset", type=float, default=0.25)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--bg_noise", type=float, default=0.0)
    # model
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    # optimization (grokking regime)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=256, help="<=0 for full-batch GD")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tag = tag_of(args)
    print(f"device={device}  tag={tag}")

    bcfg = GridConfig(grid=args.grid, cell_px=args.cell_px, train_frac=args.train_frac,
                      max_total=args.max_total, p_target=args.p_target,
                      dot_sigma=args.dot_sigma, slot_inset=args.slot_inset,
                      jitter=args.jitter, bg_noise=args.bg_noise, seed=args.seed)
    ds = make_grid_dataset(bcfg)
    train, val = split_tensors(ds)
    n_tr, n_va = len(train["images"]), len(val["images"])
    total = n_tr + n_va
    space = 6 ** bcfg.n_cells
    ib = intensity_baseline(ds["images"], ds["labels"])
    print(f"input space=6^{bcfg.n_cells}={space:.3g}  sampled={total}  train={n_tr}  val={n_va}  "
          f"classes=0..{bcfg.n_cells}  img={bcfg.img_size}px")
    print(f"intensity baseline (brightness->#targets): r2={ib['r2']:.3f} "
          f"round_acc={ib['lin_round_acc']:.3f}  majority_rate={ib['majority_rate']:.3f}")

    montage(ds, f"{FIG}/{tag}_samples.png")

    torch.manual_seed(args.seed)
    mcfg = ViTConfig(img_size=bcfg.img_size, patch_size=bcfg.cell_px, in_ch=1,
                     d_model=args.d_model, depth=args.depth, n_heads=args.n_heads,
                     n_classes=bcfg.n_classes)
    model = HookedViT(mcfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model params={nparams/1e6:.3f}M  tokens={mcfg.n_patches+1}  (patch=cell so 1 token/cell)")

    hist, best = fit_grok(model, train, val, device, steps=args.steps, bs=args.bs,
                          lr=args.lr, wd=args.wd, warmup=args.warmup,
                          eval_every=args.eval_every, seed=args.seed)
    summary = grok_summary(hist)
    print("\n=== grok summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    plot_curve(hist, summary, tag, f"{FIG}/{tag}_curve.png")
    os.makedirs(FIG, exist_ok=True)
    with open(f"{FIG}/{tag}_history.json", "w") as f:
        json.dump({"tag": tag, "args": vars(args), "n_train": n_tr, "n_val": n_va,
                   "input_space": float(space), "intensity_baseline": ib,
                   "summary": summary, "history": hist}, f, indent=2, default=float)

    os.makedirs(RUNDIR, exist_ok=True)
    ckpt = f"{RUNDIR}/{tag}.pt"
    torch.save({"model": model.state_dict(), "vit_cfg": vars(mcfg),
                "grid_cfg": vars(bcfg), "val_acc": best["val_acc"], "summary": summary}, ckpt)
    print(f"\nsaved {ckpt}")
    print(f"saved {FIG}/{tag}_curve.png")


if __name__ == "__main__":
    main()
