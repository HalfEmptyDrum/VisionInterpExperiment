"""Grokking training loop for the finite grid-counting task.

Mirrors the canonical grokking recipe (Power et al. 2022; Nanda et al. mod-113):
small memorizable train split + strong weight decay + long, constant-LR training
+ plain cross-entropy. We log train/val accuracy *and* loss at every eval so the
memorize -> generalize gap is visible: train acc -> 1.0 early while val sits at
chance, then val acc rises sharply many steps later.

``fit_grok`` is the reusable loop; ``main`` (in run.py) wires it to the data and
the HookedViT model.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def _eval(model, images, labels, device, bs=4096):
    """Return (accuracy, mean cross-entropy loss) over the given split."""
    model.eval()
    n = images.shape[0]
    correct = 0
    loss_sum = 0.0
    for s in range(0, n, bs):
        xb = images[s : s + bs].to(device)
        yb = labels[s : s + bs].to(device)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / n, loss_sum / n


def fit_grok(model, train, val, device, steps=30000, bs=256, lr=1e-3, wd=1.0,
             warmup=100, eval_every=200, betas=(0.9, 0.98), seed=0,
             log=print, keep_best=True):
    """Train ``model`` in the grokking regime; return a history list.

    Each history entry is a dict:
      {step, train_acc, train_loss, val_acc, val_loss, lr}

    ``bs <= 0`` selects full-batch gradient descent (often the cleanest grokking
    signal on tiny splits). With ``keep_best`` the model is left at its best
    val-accuracy weights at the end (so a checkpoint is analysis-ready).
    """
    Xtr, Ytr = train["images"], train["labels"]
    Xva, Yva = val["images"], val["labels"]
    n = len(Xtr)
    full_batch = bs <= 0 or bs >= n
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    g = torch.Generator().manual_seed(seed)

    hist: list[dict] = []
    best = {"val_acc": -1.0, "state": None, "step": 0}

    def record(step):
        tr_acc, tr_loss = _eval(model, Xtr, Ytr, device)
        va_acc, va_loss = _eval(model, Xva, Yva, device)
        cur_lr = opt.param_groups[0]["lr"]
        hist.append({"step": step, "train_acc": tr_acc, "train_loss": tr_loss,
                     "val_acc": va_acc, "val_loss": va_loss, "lr": cur_lr})
        if keep_best and va_acc > best["val_acc"]:
            best["val_acc"] = va_acc
            best["step"] = step
            best["state"] = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        log(f"step {step:6d}  train_acc={tr_acc:.4f} loss={tr_loss:.4f}   "
            f"val_acc={va_acc:.4f} loss={va_loss:.4f}")
        return va_acc

    record(0)
    step = 0
    while step < steps:
        model.train()
        perm = torch.arange(n) if full_batch else torch.randperm(n, generator=g)
        bsize = n if full_batch else bs
        for s in range(0, n, bsize):
            idx = perm[s : s + bsize]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            for pg in opt.param_groups:
                pg["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))  # warmup then constant
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step >= steps:
                record(step)
            if step >= steps:
                break

    if keep_best and best["state"] is not None:
        model.load_state_dict(best["state"])
    return hist, best


def grok_summary(hist, mem_thr=0.99, gen_thr=0.90, min_gap=500) -> dict:
    """Summarise a history: when train memorized, when val generalized, the gap."""
    mem = next((h["step"] for h in hist if h["train_acc"] >= mem_thr), None)
    gen = next((h["step"] for h in hist if h["val_acc"] >= gen_thr), None)
    return {
        "train_final": hist[-1]["train_acc"],
        "val_final": hist[-1]["val_acc"],
        "val_best": max(h["val_acc"] for h in hist),
        "memorize_step": mem,
        "generalize_step": gen,
        "grok_gap": (gen - mem) if (mem is not None and gen is not None) else None,
        "grokked": bool(mem is not None and gen is not None and (gen - mem) >= min_gap),
    }
