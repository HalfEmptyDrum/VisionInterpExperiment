"""Train the HookedViT to classify blob count.

    python -m mechvis.train --epochs 30

Saves the best checkpoint (by val accuracy) to ``runs/vit_count.pt`` with the
model state and the BlobConfig/ViTConfig dicts needed to rebuild for analysis.
"""
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
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
    Xtr, Ytr = train["images"], train["counts"]

    mcfg = ViTConfig(
        img_size=bcfg.img_size,
        patch_size=bcfg.patch_size,
        in_ch=1,
        d_model=args.d_model,
        depth=args.depth,
        n_heads=args.n_heads,
        n_classes=bcfg.n_classes,
    )
    model = HookedViT(mcfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = (args.n_train + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )

    g = torch.Generator().manual_seed(0)
    best = 0.0
    best_per = None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, generator=g)
        running = 0.0
        for s in range(0, args.n_train, args.bs):
            idx = perm[s : s + args.bs]
            xb = Xtr[idx].to(device)
            yb = Ytr[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * xb.size(0)
        acc, acc1, per = evaluate(model, val["images"], val["counts"], device)
        print(f"ep{ep+1:02d}  loss={running/args.n_train:.4f}  val_acc={acc:.4f}  val_acc±1={acc1:.4f}")
        if acc > best:
            best, best_per = acc, per
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "vit_cfg": vars(mcfg),
                    "blob_cfg": vars(bcfg),
                    "val_acc": acc,
                    "val_acc_pm1": acc1,
                    "per_class_acc": per,
                },
                args.out,
            )
    print(f"\nbest val_acc={best:.4f}")
    if best_per is not None:
        print("per-class acc:", [round(x, 3) for x in best_per.tolist()])


if __name__ == "__main__":
    main()
