"""Mechanistic analyses of the trained counting ViT.

    python -m mechvis.analysis --ckpt runs/vit_count.pt

Each analysis targets one of the research questions:

  Q1  patch embeddings detect blobs
      - patch_embed_filters(): visualise the conv patch projection as filters
      - patch_probe over layers: can a linear probe read local blob presence/count
        from a patch token? (high at patch_embed => detection is in the embedding)

  Q2  attention routes spatial information
      - attention_routing(): does CLS attend to blob-containing patches, and ~uniformly?
        (uniform mass on blobs == summation-by-attention)
      - head_ablation(): accuracy drop when each head's output is zeroed
      - pos_ablation(): accuracy without positional embeddings

  Q3  spatial -> abstract transition
      - cls_probe over layers: where does the global count become linearly decodable?
      - logit_lens over layers: where does the count direction align with the output?
      - patch_probe decay: where does per-patch spatial info fade?
      - activation_patching(): which layer/positions carry the movable count signal?
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from .data import BlobConfig, centers_to_patch_counts, make_dataset
from .model import HookedViT, load_checkpoint
from .viz import _to_np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "figures"


# --------------------------------------------------------------------------- #
# activation collection
# --------------------------------------------------------------------------- #
def collect(model, images, device, keys, bs=256):
    """Run the model in batches; return {key: concatenated cpu tensor} for `keys`."""
    outs = {k: [] for k in keys}
    for s in range(0, len(images), bs):
        _, c = model.run_with_cache(images[s : s + bs].to(device))
        for k in keys:
            outs[k].append(c[k].cpu())
    return {k: torch.cat(v) for k, v in outs.items()}


@torch.no_grad()
def accuracy(model, images, labels, device, hooks=None, ablate_pos=False, bs=512):
    n = len(images)
    correct = 0
    for s in range(0, n, bs):
        xb = images[s : s + bs].to(device)
        logits = model.run_with_hooks(xb, hooks=hooks, ablate_pos=ablate_pos)
        correct += (logits.argmax(1).cpu() == labels[s : s + bs]).sum().item()
    return correct / n


# --------------------------------------------------------------------------- #
# linear probes
# --------------------------------------------------------------------------- #
def _split(n, frac=0.7, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(frac * n)
    return idx[:k], idx[k:]


def _zscore_fit(X, tr):
    """Standardize features using train-split statistics (helps probe convergence)."""
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    return (X - mu) / sd


def probe_clf(X, y, seed=0):
    """Multinomial logistic probe -> held-out accuracy."""
    X = _to_np(X).astype(np.float64)
    y = _to_np(y).astype(np.int64)
    tr, te = _split(len(X), seed=seed)
    X = _zscore_fit(X, tr)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X[tr], y[tr])
    return float(clf.score(X[te], y[te]))


def probe_reg_r2(X, y, seed=0):
    """Ridge probe -> held-out R^2 (for ordinal/continuous targets)."""
    X = _to_np(X).astype(np.float64)
    y = _to_np(y).astype(np.float64)
    tr, te = _split(len(X), seed=seed)
    X = _zscore_fit(X, tr)
    reg = Ridge(alpha=1.0).fit(X[tr], y[tr])
    return float(reg.score(X[te], y[te]))


def probe_auc(X, y, seed=0):
    """Balanced logistic probe -> held-out ROC-AUC (robust to class imbalance)."""
    X = _to_np(X).astype(np.float64)
    y = _to_np(y).astype(np.int64)
    tr, te = _split(len(X), seed=seed)
    X = _zscore_fit(X, tr)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], clf.decision_function(X[te])))


def cls_points(depth):
    """Ordered residual-stream points (CLS) from input to output."""
    pts = ["embed"]
    for i in range(depth):
        pts += [f"blocks.{i}.resid_mid", f"blocks.{i}.resid_post"]
    pts += ["resid_final"]
    return pts


def run_probes(model, ds, device, n_sub=1500):
    depth = model.cfg.depth
    imgs = ds["images"][:n_sub]
    counts = ds["counts"][:n_sub]
    # recompute per-patch labels at the *model's* patch grid (robust to patch_size sweeps)
    pcounts = torch.from_numpy(np.stack([
        centers_to_patch_counts(ds["centers"][i], model.cfg.img_size, model.cfg.patch_size)
        for i in range(n_sub)]))  # (N, n_patches)

    cls_keys = cls_points(depth)
    patch_keys = ["patch_embed"] + [f"blocks.{i}.resid_post" for i in range(depth)]
    acts = collect(model, imgs, device, list(dict.fromkeys(cls_keys + patch_keys)))

    # Q3: global count decodable from CLS, layer by layer
    cls_acc = {k: probe_clf(acts[k][:, 0], counts) for k in cls_keys}

    # Q3: logit-lens count accuracy from CLS at each layer
    lens = logit_lens(model, acts, counts)

    # Q1/Q3: local blob presence/count decodable from a patch token, layer by layer.
    # Presence is ~94% empty, so we use AUC (not accuracy) + a subsample for speed.
    presence = (pcounts > 0).reshape(-1).numpy()
    local_count = pcounts.reshape(-1).numpy()
    nrow = presence.shape[0]
    sel = np.random.default_rng(0).choice(nrow, min(nrow, 40000), replace=False)
    patch_auc, patch_r2 = {}, {}
    for k in patch_keys:
        # patch_embed has no CLS (n_patches tokens); resid_* has CLS at 0 (n_patches+1).
        # Taking the last n_patches tokens selects patch tokens correctly for both.
        Xp = acts[k][:, -model.cfg.n_patches:, :].reshape(nrow, model.cfg.d_model).numpy()[sel]
        patch_auc[k] = probe_auc(Xp, presence[sel])
        patch_r2[k] = probe_reg_r2(Xp, local_count[sel])
    return {"cls_acc": cls_acc, "logit_lens": lens, "patch_presence_auc": patch_auc,
            "patch_count_r2": patch_r2}


def logit_lens(model, acts, counts):
    """Apply final LN + head to the CLS residual at each layer; report accuracy."""
    out = {}
    counts_np = _to_np(counts)
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(model.cfg.depth):
            for which in ("resid_mid", "resid_post"):
                key = f"blocks.{i}.{which}"
                cls = acts[key][:, 0].to(dev)
                pred = model.head(model.ln_f(cls)).argmax(1).cpu().numpy()
                out[key] = float((pred == counts_np).mean())
        cls = acts["resid_final"][:, 0].to(dev)  # already LN'd
        out["resid_final"] = float((model.head(cls).argmax(1).cpu().numpy() == counts_np).mean())
    return out


# --------------------------------------------------------------------------- #
# Q1: patch-embedding filters
# --------------------------------------------------------------------------- #
def patch_embed_filters(model, path=f"{FIG}/patch_filters.png", n=32):
    W = model.patch.weight.detach().cpu()  # (d_model, in_ch, P, P)
    n = min(n, W.shape[0])
    ncol = 8
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.1, nrow * 1.1))
    for j, ax in enumerate(np.atleast_1d(axes).flat):
        if j < n:
            f = W[j, 0]
            v = f.abs().max().item() or 1e-6
            ax.imshow(f, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.axis("off")
    fig.suptitle("patch-embedding filters (conv weights)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Q2: attention routing
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_routing(model, ds, device, n_sub=512):
    imgs = ds["images"][:n_sub].to(device)
    pc = np.stack([centers_to_patch_counts(ds["centers"][i], model.cfg.img_size, model.cfg.patch_size)
                   for i in range(n_sub)])
    presence = (torch.from_numpy(pc) > 0).float()  # (N, n_patches) at model grid
    base_rate = presence.mean().item()  # fraction of patches that contain a blob
    _, cache = model.run_with_cache(imgs)
    depth, H = model.cfg.depth, model.cfg.n_heads
    rows = []
    for i in range(depth):
        pat = cache[f"blocks.{i}.attn.pattern"].cpu()  # (N, H, 65, 65)
        for h in range(H):
            cls_attn = pat[:, h, 0, 1:]  # (N, 64) CLS -> patches
            mass_on_blobs = (cls_attn * presence).sum(1) / cls_attn.sum(1).clamp(min=1e-9)
            # uniformity of attention *within* blob patches (CV; low => ~uniform == sum)
            cv = []
            for img_attn, pres in zip(cls_attn, presence):
                w = img_attn[pres > 0]
                if len(w) >= 2:
                    cv.append((w.std() / w.mean().clamp(min=1e-9)).item())
            rows.append({
                "layer": i, "head": h,
                "cls_mass_on_blobs": float(mass_on_blobs.mean()),
                "blob_base_rate": base_rate,
                "within_blob_cv": float(np.mean(cv)) if cv else float("nan"),
                "entropy": float(_entropy(pat[:, h, 0]).mean()),
            })
    return rows


def _entropy(p):  # p: (N, T) attention distribution
    return -(p.clamp(min=1e-12) * p.clamp(min=1e-12).log()).sum(-1)


@torch.no_grad()
def attention_examples(model, ds, device, path=f"{FIG}/attn_examples.png", n=4):
    imgs = ds["images"][:n].to(device)
    _, cache = model.run_with_cache(imgs)
    depth, H = model.cfg.depth, model.cfg.n_heads
    grid = model.cfg.img_size // model.cfg.patch_size
    fig, axes = plt.subplots(n, 1 + depth * H, figsize=((1 + depth * H) * 1.3, n * 1.3))
    axes = np.atleast_2d(axes)
    for r in range(n):
        ax = axes[r, 0]
        im = _to_np(imgs[r, 0])
        ax.imshow(im, cmap="magma", vmin=0, vmax=max(im.max(), 1e-6))
        ax.set_title(f"k={int(ds['counts'][r])}", fontsize=8)
        ax.axis("off")
        col = 1
        for i in range(depth):
            pat = cache[f"blocks.{i}.attn.pattern"].cpu()
            for h in range(H):
                a = pat[r, h, 0, 1:].reshape(grid, grid)
                ax = axes[r, col]
                ax.imshow(a, cmap="viridis")
                if r == 0:
                    ax.set_title(f"L{i}H{h}", fontsize=7)
                ax.axis("off")
                col += 1
    fig.suptitle("CLS->patch attention (col 0 = input)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Q2: ablations
# --------------------------------------------------------------------------- #
def head_ablation(model, ds, device):
    imgs, labels = ds["images"], ds["counts"]
    base = accuracy(model, imgs, labels, device)
    rows = []
    for i in range(model.cfg.depth):
        for h in range(model.cfg.n_heads):
            hooks = {f"blocks.{i}.attn.z": _zero_head(h)}
            acc = accuracy(model, imgs, labels, device, hooks=hooks)
            rows.append({"layer": i, "head": h, "acc": acc, "drop": base - acc})
    return base, rows


def _zero_head(h):
    def fn(z):  # z: (B, T, H, dh)
        z = z.clone()
        z[:, :, h, :] = 0
        return z
    return fn


def pos_ablation(model, ds, device):
    base = accuracy(model, ds["images"], ds["counts"], device)
    no_pos = accuracy(model, ds["images"], ds["counts"], device, ablate_pos=True)
    return base, no_pos


# --------------------------------------------------------------------------- #
# Q3: activation patching
# --------------------------------------------------------------------------- #
@torch.no_grad()
def activation_patching(model, ds, device, n_pairs=256):
    """Patch resid_post from a source image into a target, sweeping layer & site.

    Reports the mean shift of the predicted count toward the source's count
    (1.0 == fully takes on the source count; 0.0 == unchanged).
    """
    imgs = ds["images"][:n_pairs * 2].to(device)
    counts = ds["counts"][:n_pairs * 2]
    tgt, src = imgs[:n_pairs], imgs[n_pairs:]
    ct, cs = counts[:n_pairs].float(), counts[n_pairs:].float()

    base_pred = model.run_with_hooks(tgt).argmax(1).cpu().float()
    _, src_cache = model.run_with_cache(src)

    depth = model.cfg.depth
    results = {}
    for site, sl in (("patches", slice(1, None)), ("cls", slice(0, 1))):
        row = []
        for i in range(depth):
            src_resid = src_cache[f"blocks.{i}.resid_post"].to(device)
            hooks = {f"blocks.{i}.resid_post": _patch_positions(src_resid, sl)}
            pred = model.run_with_hooks(tgt, hooks=hooks).argmax(1).cpu().float()
            # normalised shift toward source count
            denom = (cs - ct).abs().clamp(min=1e-6)
            shift = ((pred - base_pred) / (cs - ct).where((cs - ct).abs() > 0, denom)).clamp(-1, 2)
            row.append({"layer": i, "shift_to_src": float(shift.mean()),
                        "pred_changed": float((pred != base_pred).float().mean())})
        results[site] = row
    return results


def _patch_positions(src_resid, sl):
    def fn(x):
        x = x.clone()
        x[:, sl, :] = src_resid[:, sl, :]
        return x
    return fn


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def plot_trajectory(probes, path=f"{FIG}/probe_trajectory.png"):
    pts = list(probes["cls_acc"].keys())
    x = range(len(pts))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, [probes["cls_acc"][p] for p in pts], "o-", label="CLS count probe")
    ax.plot(x, [probes["logit_lens"].get(p, np.nan) for p in pts], "s--", label="logit lens")
    pp = probes["patch_presence_auc"]
    px = [pts.index(k) if k in pts else None for k in pp]
    ax.plot([i for i in px if i is not None],
            [pp[k] for k, i in zip(pp, px) if i is not None],
            "^:", label="patch presence AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pts, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("decodability across depth (spatial -> abstract)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/vit_count.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device

    model, ckpt = load_checkpoint(args.ckpt, device)
    print(f"loaded {args.ckpt}  val_acc={ckpt.get('val_acc'):.4f}  depth={model.cfg.depth} heads={model.cfg.n_heads}")
    bcfg = BlobConfig(**ckpt["blob_cfg"])
    ds = make_dataset(bcfg, args.n, seed=123)
    os.makedirs(FIG, exist_ok=True)

    print("\n[Q1/Q3] linear probes ...")
    probes = run_probes(model, ds, device)
    for p, a in probes["cls_acc"].items():
        print(f"  CLS-count  {p:24s} acc={a:.3f}   lens={probes['logit_lens'].get(p, float('nan')):.3f}")
    print("  -- patch-token local blob info --")
    for p in probes["patch_presence_auc"]:
        print(f"  patch      {p:24s} presence_AUC={probes['patch_presence_auc'][p]:.3f}  count_R2={probes['patch_count_r2'][p]:.3f}")
    print("  figure:", plot_trajectory(probes))
    print("  figure:", patch_embed_filters(model))

    print("\n[Q2] attention routing (CLS mass on blob patches; base rate ~%.3f) ..." % (
        (ds["patch_counts"][:512] > 0).float().mean().item()))
    for r in attention_routing(model, ds, device):
        print(f"  L{r['layer']}H{r['head']}  mass_on_blobs={r['cls_mass_on_blobs']:.3f}  "
              f"within_blob_cv={r['within_blob_cv']:.2f}  entropy={r['entropy']:.2f}")
    print("  figure:", attention_examples(model, ds, device))

    print("\n[Q2] head ablation (accuracy drop) ...")
    base, rows = head_ablation(model, ds, device)
    print(f"  baseline acc = {base:.3f}")
    for r in sorted(rows, key=lambda d: -d["drop"]):
        print(f"  L{r['layer']}H{r['head']}  acc={r['acc']:.3f}  drop={r['drop']:+.3f}")
    pbase, no_pos = pos_ablation(model, ds, device)
    print(f"  positional-embedding ablation: {pbase:.3f} -> {no_pos:.3f}  (drop {pbase-no_pos:+.3f})")

    print("\n[Q3] activation patching (shift of predicted count toward source) ...")
    patch_res = activation_patching(model, ds, device)
    for site, rows in patch_res.items():
        for r in rows:
            print(f"  patch {site:7s} @ blocks.{r['layer']}.resid_post  "
                  f"shift_to_src={r['shift_to_src']:+.3f}  changed={r['pred_changed']:.3f}")

    summary = {"val_acc": ckpt.get("val_acc"), "probes": probes,
               "patching": patch_res}
    with open(f"{FIG}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {FIG}/metrics.json")


if __name__ == "__main__":
    main()
