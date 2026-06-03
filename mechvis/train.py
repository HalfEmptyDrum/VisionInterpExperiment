"""Train the HookedViT to classify blob count.

    python -m mechvis.train --epochs 40 --n_train 40000

Saves the best checkpoint (by val accuracy) to ``runs/vit_count.pt`` with the
model state and the BlobConfig/ViTConfig dicts needed to rebuild for analysis.
``fit()`` is the reusable training loop (also used by the architecture sweep).
"""
from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.nn.functional as F

from .data import BlobConfig, make_dataset
from .model import HookedViT, ViTConfig


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, images, labels, device, bs=512):
    model.eval()
    n = images.shape[0]
    C = model.cfg.n_classes
    correct = 0
    within1 = 0
    per_correct = torch.zeros(C)
    per_total = torch.zeros(C)
    for s in range(0, n, bs):
        xb = images[s : s + bs].to(device)
        yb = labels[s : s + bs].cpu()
        pred = model(xb).argmax(1).cpu()
        hit = pred == yb
        correct += hit.sum().item()
        within1 += (pred - yb).abs().le(1).sum().item()
        for c in range(C):
            m = yb == c
            per_total[c] += m.sum().item()
            per_correct[c] += (hit & m).sum().item()
    return correct / n, within1 / n, (per_correct / per_total.clamp(min=1))


def fit(model, train, val, device, epochs=40, bs=256, lr=3e-4, wd=0.01,
        label_smoothing=0.1, seed=0, verbose=True):
    """Train ``model`` in place; leave it at its best-val-accuracy weights.

    Returns a dict: {acc, acc1, per_class, history}.
    """
    Xtr, Ytr = train["images"], train["counts"]
    n_train = len(Xtr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((n_train + bs - 1) // bs)
    # OneCycle needs enough steps to define its warmup; fall back to constant LR for tiny runs
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
             if steps >= 20 else None)
    g = torch.Generator().manual_seed(seed)

    best = {"acc": -1.0}
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=g)
        running = 0.0
        for s in range(0, n_train, bs):
            idx = perm[s : s + bs]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            running += loss.item() * xb.size(0)
        acc, acc1, per = evaluate(model, val["images"], val["counts"], device)
        history.append((running / n_train, acc, acc1))
        if verbose:
            print(f"ep{ep+1:02d}  loss={running/n_train:.4f}  val_acc={acc:.4f}  val_acc±1={acc1:.4f}")
        if acc > best["acc"]:
            best = {"acc": acc, "acc1": acc1, "per_class": per,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
    model.load_state_dict(best.pop("state"))  # restore best weights
    best["history"] = history
    return best


@torch.no_grad()
def _acc_only(model, images, labels, device, bs=1024):
    model.eval()
    correct = 0
    for s in range(0, len(images), bs):
        pred = model(images[s : s + bs].to(device)).argmax(1).cpu()
        correct += (pred == labels[s : s + bs]).sum().item()
    return correct / len(images)


def fit_grok(model, train, val, device, steps=12000, bs=256, lr=1e-3, wd=1.0,
             warmup=200, eval_every=200, seed=0):
    """Grokking regime: small memorizable train set + strong weight decay + long
    constant-LR training + plain CE. Logs (step, train_acc, val_acc) so the
    memorize->generalize gap is visible. Returns the history list.
    """
    Xtr, Ytr = train["images"], train["counts"]
    n = len(Xtr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    g = torch.Generator().manual_seed(seed)
    hist = []
    step = 0
    while step < steps:
        model.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            for pg in opt.param_groups:
                pg["lr"] = lr * min(1.0, (step + 1) / warmup)  # warmup then constant
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step == 1 or step % eval_every == 0 or step >= steps:
                hist.append((step, _acc_only(model, Xtr, Ytr, device),
                             _acc_only(model, val["images"], val["counts"], device)))
            if step >= steps:
                break
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--out", type=str, default="runs/vit_count.pt")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(0)
    print(f"device={device}")

    bcfg = BlobConfig()
    print("generating data ...")
    train = make_dataset(bcfg, args.n_train, seed=1)
    val = make_dataset(bcfg, args.n_val, seed=2)

    mcfg = ViTConfig(
        img_size=bcfg.img_size, patch_size=args.patch_size, in_ch=1,
        d_model=args.d_model, depth=args.depth, n_heads=args.n_heads, n_classes=bcfg.n_classes,
    )
    model = HookedViT(mcfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.3f}M  tokens={mcfg.n_patches+1}")

    best = fit(model, train, val, device, epochs=args.epochs, bs=args.bs, lr=args.lr,
               wd=args.wd, label_smoothing=args.label_smoothing)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "vit_cfg": vars(mcfg), "blob_cfg": vars(bcfg),
                "val_acc": best["acc"], "val_acc_pm1": best["acc1"],
                "per_class_acc": best["per_class"]}, args.out)
    print(f"\nbest val_acc={best['acc']:.4f}  val_acc±1={best['acc1']:.4f}")
    print("per-class acc:", [round(x, 3) for x in best["per_class"].tolist()])
    print("saved", args.out)


if __name__ == "__main__":
    main()
