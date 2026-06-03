"""Unattended overnight experiment.

Thorough architecture sweep (accuracy + interpretability) -> multi-seed
replication of the top configs -> full mechanistic analysis on the winner ->
REPORT.md. Designed to run for hours without supervision:

* HARD TIME BUDGET. Every training run is guarded against a deadline; if a run
  would not finish in time it is skipped. ~20 min is always reserved at the end
  so the final analysis + report run no matter what.
* CRASH-RESILIENT. Each run is isolated in try/except; results are written to
  figures/overnight_results.json after every run; GPU memory is freed between
  runs. Re-launching resumes (skips already-finished tag+seed).

    python -u -m mechvis.overnight --budget_hours 6

Notes on hardware: patch_size=4 (257 tokens) thrashes MPS unified memory at
bs=256, so patch-4 runs use bs=128. patch-8 (65 tokens) uses bs=256.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import traceback

import numpy as np
import torch

from . import analysis as A
from .data import BlobConfig, make_dataset
from .model import HookedViT, ViTConfig, load_checkpoint
from .sweep import _minmax, _pareto, interp_metrics, plot_results
from .train import fit, fit_grok, pick_device

RUNDIR = "runs/sweep"
FIG = "figures"
RESULTS = f"{FIG}/overnight_results.json"
GROKJSON = f"{FIG}/grok_results.json"
START = time.time()


def log(msg):
    el = time.time() - START
    print(f"[{int(el//3600)}h{int((el%3600)//60):02d}m] {msg}", flush=True)


def bs_for(patch):
    return 128 if patch == 4 else 256


def _dev_factor():
    # CUDA is ~20x faster than MPS for these shapes; shrink estimates so the
    # deadline guard does not wrongly skip runs on GPU.
    return 0.05 if torch.cuda.is_available() else 1.0


def est_seconds(patch, depth, width, n_train, epochs):
    """Conservative wall-time estimate (s), calibrated to observed throughput."""
    a, wf = (31.0, {64: 0.63}) if patch == 8 else (150.0, {64: 0.7})  # observed on MPS
    spe40 = a * depth * wf.get(width, width / 128.0)
    return spe40 * (n_train / 40000.0) * epochs * 1.25 * _dev_factor()


def tag_of(patch, depth, width):
    return f"p{patch}d{depth}w{width}"


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return []


def save_results(rows):
    os.makedirs(FIG, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(rows, f, indent=2, default=float)


def run_one(patch, depth, width, epochs, n_train, seed, data, deadline, reserve, results):
    tag = tag_of(patch, depth, width)
    if any(r["tag"] == tag and r["seed"] == seed for r in results):
        log(f"skip {tag} s{seed} (already done)")
        return
    est = est_seconds(patch, depth, width, n_train, epochs)
    if time.time() + est > deadline - reserve:
        log(f"SKIP {tag} s{seed}: est {est/60:.0f}min exceeds remaining budget")
        return
    train, val, ana = data
    device = pick_device("auto")
    model = None
    try:
        torch.manual_seed(seed)
        mcfg = ViTConfig(img_size=64, patch_size=patch, in_ch=1, d_model=width,
                         depth=depth, n_heads=4, n_classes=9)
        model = HookedViT(mcfg).to(device)
        n_tr = min(n_train, len(train["images"]))
        sub = {"images": train["images"][:n_tr], "counts": train["counts"][:n_tr]}
        t0 = time.time()
        best = fit(model, sub, val, device, epochs=epochs, bs=bs_for(patch), seed=seed, verbose=False)
        dt = time.time() - t0
        im = interp_metrics(model, ana, device)
        row = {"tag": tag, "patch_size": patch, "depth": depth, "d_model": width,
               "n_tokens": mcfg.n_patches + 1, "params": sum(p.numel() for p in model.parameters()),
               "seed": seed, "epochs": epochs, "n_train": n_tr,
               "acc": best["acc"], "acc1": best["acc1"],
               "per_class": [round(x, 3) for x in best["per_class"].tolist()],
               "s_per_epoch": round(dt / epochs, 1), **im}
        os.makedirs(RUNDIR, exist_ok=True)
        ckpt = f"{RUNDIR}/{tag}_s{seed}.pt"
        torch.save({"model": model.state_dict(), "vit_cfg": vars(mcfg),
                    "blob_cfg": vars(BlobConfig()), "val_acc": best["acc"]}, ckpt)
        row["ckpt"] = ckpt
        results.append(row)
        save_results(results)
        log(f"DONE {tag} s{seed}: acc={row['acc']:.3f} acc1={row['acc1']:.3f} "
            f"detect={im['detect_auc']:.3f} route={im['route_sel']:.3f} "
            f"abstract={im['acc_abstract']:.3f} | {dt/60:.1f}min ({row['s_per_epoch']}s/ep)")
    except Exception:
        log(f"FAIL {tag} s{seed}:\n{traceback.format_exc()}")
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def grok_summary(hist):
    mem = next((s for s, tr, va in hist if tr >= 0.95), None)
    gen = next((s for s, tr, va in hist if va >= 0.60), None)
    return {"train_final": hist[-1][1], "val_final": hist[-1][2],
            "val_best": max(va for _, _, va in hist),
            "memorize_step": mem, "generalize_step": gen,
            "grok_gap": (gen - mem) if (mem and gen) else None,
            "grokked": bool(mem and gen and (gen - mem) >= 1000)}


def run_grok(n_train, wd, lr, steps, data, deadline, reserve, groks, depth=2, width=64):
    tag = f"grok_n{n_train}_wd{wd}_d{depth}w{width}"
    if any(g["tag"] == tag for g in groks):
        log(f"skip {tag} (already done)")
        return
    est = steps * 0.45 * 1.2 * _dev_factor()  # ~0.3-0.45 s/step on MPS; far less on CUDA
    if time.time() + est > deadline - reserve:
        log(f"SKIP {tag}: est {est/60:.0f}min exceeds remaining budget")
        return
    train, val, _ = data
    device = pick_device("auto")
    model = None
    try:
        torch.manual_seed(0)
        mcfg = ViTConfig(img_size=64, patch_size=8, in_ch=1, d_model=width, depth=depth, n_heads=4, n_classes=9)
        model = HookedViT(mcfg).to(device)
        sub = {"images": train["images"][:n_train], "counts": train["counts"][:n_train]}
        t0 = time.time()
        hist = fit_grok(model, sub, val, device, steps=steps, wd=wd, lr=lr)
        dt = time.time() - t0
        s = grok_summary(hist)
        groks.append({"tag": tag, "n_train": n_train, "wd": wd, "lr": lr, "steps": steps,
                      "history": hist, **s})
        os.makedirs(FIG, exist_ok=True)
        with open(GROKJSON, "w") as f:
            json.dump(groks, f, default=float)
        log(f"DONE {tag}: train {s['train_final']:.2f} val {s['val_final']:.2f} "
            f"best {s['val_best']:.2f} grokked={s['grokked']} gap={s['grok_gap']} | {dt/60:.1f}min")
    except Exception:
        log(f"FAIL {tag}:\n{traceback.format_exc()}")
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def grok_plot(groks, path=f"{FIG}/grok_curves.png"):
    if not groks:
        return None
    import matplotlib.pyplot as plt
    k = len(groks)
    cols = min(k, 3)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), squeeze=False)
    for ax, g in zip(axes.flat, groks):
        h = g["history"]
        xs = [p[0] for p in h]
        ax.plot(xs, [p[1] for p in h], color="tab:blue", label="train")
        ax.plot(xs, [p[2] for p in h], color="tab:red", label="val")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step")
        ax.set_ylabel("accuracy")
        ax.set_title(f"n={g['n_train']} wd={g['wd']} lr={g['lr']}", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes.flat[k:]:
        ax.axis("off")
    fig.suptitle("grokking: train vs val accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def aggregate(results):
    """Mean over seeds per config; attach mechanism_clarity + combined."""
    by_tag = {}
    for r in results:
        by_tag.setdefault(r["tag"], []).append(r)
    rows = []
    for tag, rs in by_tag.items():
        m = rs[0]
        rows.append({
            "tag": tag, "patch_size": m["patch_size"], "depth": m["depth"],
            "d_model": m["d_model"], "n_tokens": m["n_tokens"], "params": m["params"],
            "n_seeds": len(rs),
            "acc": float(np.mean([r["acc"] for r in rs])),
            "acc_std": float(np.std([r["acc"] for r in rs])),
            "acc1": float(np.mean([r["acc1"] for r in rs])),
            "detect_auc": float(np.mean([r["detect_auc"] for r in rs])),
            "route_sel": float(np.mean([r["route_sel"] for r in rs])),
            "acc_abstract": float(np.mean([r["acc_abstract"] for r in rs])),
            "transition_frac": float(np.mean([r["transition_frac"] for r in rs])),
            "best_ckpt": min(rs, key=lambda r: -r["acc"]).get("ckpt"),
        })
    nd, nr, na = (_minmax([r["detect_auc"] for r in rows]), _minmax([r["route_sel"] for r in rows]),
                  _minmax([r["acc_abstract"] for r in rows]))
    acc_n = _minmax([r["acc"] for r in rows])
    for i, r in enumerate(rows):
        r["mechanism_clarity"] = float(np.mean([nd[i], nr[i], na[i]]))
        r["combined"] = float(0.6 * acc_n[i] + 0.4 * r["mechanism_clarity"])
    front = set(_pareto([r["acc"] for r in rows], [r["mechanism_clarity"] for r in rows]))
    for i, r in enumerate(rows):
        r["pareto"] = i in front
    return rows


def _img(path, max_w="100%"):
    if not os.path.exists(path):
        return f'<p class="missing">missing figure: {path}</p>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img style="max-width:{max_w}" src="data:image/png;base64,{b64}"/>'


REPORT_CSS = """
:root{--bg:#0f1117;--card:#1a1d27;--ink:#e7e9ee;--mut:#9aa0ad;--acc:#6ea8fe;--good:#3ddc97;--warn:#ffb454;--line:#2a2e3a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:40px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:16px;color:var(--acc);margin:26px 0 8px}
.meta{color:var(--mut);margin:0 0 20px}
code{background:#11141c;padding:1px 6px;border-radius:5px;font-size:13px;color:#cdd3e0}
.cards{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px;flex:1;min-width:240px}
.card .k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:20px;font-weight:650;margin:4px 0;font-family:ui-monospace,Menlo,monospace;color:var(--good)}
.card .s{color:var(--mut);font-size:13px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;background:var(--bg)}
td:first-child,th:first-child{text-align:left}
tr.pareto{background:rgba(61,220,151,.09)}
td.tag{font-family:ui-monospace,Menlo,monospace}
.badge{display:inline-block;font-size:10.5px;padding:1px 7px;border-radius:20px;margin-left:7px;vertical-align:middle}
.badge.comb{background:rgba(110,168,254,.18);color:var(--acc);border:1px solid var(--acc)}
.badge.acc{background:rgba(61,220,151,.16);color:var(--good);border:1px solid var(--good)}
figure{margin:14px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
figure img{display:block;border-radius:6px;background:#fff}
figcaption{color:var(--mut);font-size:12.5px;margin-top:8px}
.missing{color:var(--warn)}
.note{color:var(--mut);font-size:13px;margin:6px 0 4px}
"""


def write_report(agg, full, device_name, groks):
    rt = (time.time() - START) / 3600
    if not agg:  # pathological: every sweep run skipped/failed -> still emit a (grok-only) report
        best_acc = best_comb = {"tag": "n/a", "acc": float("nan"), "acc_std": float("nan"),
                                "mechanism_clarity": float("nan")}
    else:
        best_acc = max(agg, key=lambda r: r["acc"])
        best_comb = max(agg, key=lambda r: r["combined"])

    trs = ""
    for r in sorted(agg, key=lambda r: -r["combined"]):
        badge = ('<span class="badge comb">best combined</span>' if r is best_comb else "")
        badge += ('<span class="badge acc">best acc</span>' if r is best_acc else "")
        trs += (f'<tr class="{"pareto" if r["pareto"] else ""}">'
                f'<td class="tag">{r["tag"]}{badge}</td><td>{r["n_tokens"]}</td>'
                f'<td>{r["params"]/1e6:.2f}M</td><td>{r["acc"]:.3f}</td><td>±{r["acc_std"]:.3f}</td>'
                f'<td>{r["acc1"]:.3f}</td><td>{r["detect_auc"]:.3f}</td><td>{r["route_sel"]:.3f}</td>'
                f'<td>{r["acc_abstract"]:.3f}</td><td><b>{r["mechanism_clarity"]:.3f}</b></td>'
                f'<td><b>{r["combined"]:.3f}</b></td><td>{r["n_seeds"]}</td>'
                f'<td>{"★" if r["pareto"] else ""}</td></tr>')
    sweep_tbl = ('<table><thead><tr><th>config</th><th>tok</th><th>params</th><th>acc</th><th>std</th>'
                 '<th>±1</th><th>detect</th><th>route</th><th>abstract</th><th>clarity</th>'
                 '<th>combined</th><th>seeds</th><th>P</th></tr></thead><tbody>' + trs + '</tbody></table>')

    body = (
        f'<div class="wrap"><h1>MechInterpVision — overnight report</h1>'
        f'<p class="meta">Generated after {rt:.1f} h on <code>{device_name}</code>. '
        f'Task: count Gaussian blobs (0–8) in 64×64 images. The intensity baseline '
        f'(summed pixel intensity → count) is at chance, so the model must genuinely detect and count.</p>'
        f'<div class="cards">'
        f'<div class="card"><div class="k">best accuracy</div><div class="v">{best_acc["tag"]}</div>'
        f'<div class="s">{best_acc["acc"]:.3f} ± {best_acc["acc_std"]:.3f} exact</div></div>'
        f'<div class="card"><div class="k">best accuracy + interpretability</div><div class="v">{best_comb["tag"]}</div>'
        f'<div class="s">acc {best_comb["acc"]:.3f} · clarity {best_comb["mechanism_clarity"]:.3f}</div></div>'
        f'</div>'
        f'<h2>Architecture sweep</h2>'
        f'<p class="note">Mean over seeds. <b>clarity</b> = mean of normalised detect / route / abstract proxies; '
        f'<b>combined</b> = 0.6·norm(acc) + 0.4·clarity. Highlighted rows are Pareto-optimal '
        f'(nothing beats them on both accuracy and clarity).</p>{sweep_tbl}'
        f'<figure>{_img("figures/sweep_scatter.png")}<figcaption>Accuracy vs interpretability — point size '
        f'∼ tokens; blue = patch-8, red = patch-4; outlined = Pareto frontier.</figcaption></figure>'
        f'<figure>{_img("figures/sweep_heatmap.png")}<figcaption>Exact accuracy across depth × patch, per width.</figcaption></figure>'
    )

    if full:
        q3 = '<table><thead><tr><th>residual point</th><th>count-probe acc</th><th>logit-lens acc</th></tr></thead><tbody>'
        for k, a in full["cls_acc"].items():
            q3 += f'<tr><td class="tag">{k}</td><td>{a:.3f}</td><td>{full["logit_lens"].get(k, float("nan")):.3f}</td></tr>'
        q3 += '</tbody></table>'
        q1 = '<table><thead><tr><th>layer</th><th>presence AUC</th><th>local-count R²</th></tr></thead><tbody>'
        for k in full["patch_presence_auc"]:
            q1 += (f'<tr><td class="tag">{k}</td><td>{full["patch_presence_auc"][k]:.3f}</td>'
                   f'<td>{full["patch_count_r2"][k]:.3f}</td></tr>')
        q1 += '</tbody></table>'
        q2 = ('<table><thead><tr><th>head</th><th>CLS mass on blobs</th><th>within-blob CV</th>'
              '<th>entropy</th></tr></thead><tbody>')
        for r in full["routing"]:
            q2 += (f'<tr><td class="tag">L{r["layer"]}H{r["head"]}</td><td>{r["cls_mass_on_blobs"]:.3f}</td>'
                   f'<td>{r["within_blob_cv"]:.2f}</td><td>{r["entropy"]:.2f}</td></tr>')
        q2 += '</tbody></table>'
        hab = '<table><thead><tr><th>head</th><th>acc drop when zeroed</th></tr></thead><tbody>'
        for r in sorted(full["head_ablation"], key=lambda d: -d["drop"])[:8]:
            hab += f'<tr><td class="tag">L{r["layer"]}H{r["head"]}</td><td>{r["drop"]:+.3f}</td></tr>'
        hab += '</tbody></table>'
        pat = '<table><thead><tr><th>site @ layer</th><th>shift to source</th><th>frac changed</th></tr></thead><tbody>'
        for site, rs in full["patching"].items():
            for r in rs:
                pat += (f'<tr><td class="tag">{site} @ blocks.{r["layer"]}</td>'
                        f'<td>{r["shift_to_src"]:+.3f}</td><td>{r["pred_changed"]:.3f}</td></tr>')
        pat += '</tbody></table>'
        body += (
            f'<h2>Mechanistic analysis of <code>{full["tag"]}</code></h2>'
            f'<p class="note">Best checkpoint (val acc {full["val_acc"]:.3f}). Blob base rate '
            f'≈ {full["base_rate"]:.3f} (fraction of patches containing a centre).</p>'
            f'<h3>Q1 — patch embeddings detect blobs</h3>'
            f'<p class="note">Linear probe for local blob presence / count from one patch token. '
            f'High already at <code>patch_embed</code> ⇒ detection lives in the embedding.</p>{q1}'
            f'<figure>{_img("figures/patch_filters.png")}<figcaption>Patch-projection conv weights as filters.</figcaption></figure>'
            f'<h3>Q2 — attention routes spatial information</h3>'
            f'<p class="note">CLS→patch attention. Mass above base rate ⇒ routing to objects; '
            f'low within-blob CV ⇒ ~uniform weighting (summation).</p>{q2}'
            f'<p class="note">Positional-embedding ablation: {full["pos_base"]:.3f} → {full["pos_nopos"]:.3f} '
            f'(drop {full["pos_base"]-full["pos_nopos"]:+.3f}). Per-head ablations:</p>{hab}'
            f'<figure>{_img("figures/attn_examples.png")}<figcaption>CLS→patch attention per head (col 0 = input image).</figcaption></figure>'
            f'<h3>Q3 — spatial→abstract transition</h3>'
            f'<p class="note">Where the global count becomes linearly decodable from CLS, by depth.</p>{q3}'
            f'<figure>{_img("figures/probe_trajectory.png")}<figcaption>Decodability across depth.</figcaption></figure>'
            f'<p class="note">Activation patching: overwrite residual positions from a source image; shift = '
            f'how far the predicted count moves toward the source count (1.0 = fully).</p>{pat}'
        )
    else:
        body += '<h2>Mechanistic analysis</h2><p class="missing">Did not run (no winner checkpoint).</p>'

    if groks:
        gt = ('<table><thead><tr><th>config</th><th>wd</th><th>n_train</th><th>train acc</th>'
              '<th>val acc</th><th>val best</th><th>memorize@</th><th>generalize@</th>'
              '<th>gap</th><th>grokked</th></tr></thead><tbody>')
        for g in groks:
            gt += (f'<tr class="{"pareto" if g["grokked"] else ""}"><td class="tag">{g["tag"]}</td>'
                   f'<td>{g["wd"]}</td><td>{g["n_train"]}</td><td>{g["train_final"]:.3f}</td>'
                   f'<td>{g["val_final"]:.3f}</td><td>{g["val_best"]:.3f}</td>'
                   f'<td>{g["memorize_step"]}</td><td>{g["generalize_step"]}</td>'
                   f'<td>{g["grok_gap"]}</td><td>{"yes" if g["grokked"] else "no"}</td></tr>')
        gt += '</tbody></table>'
        body += (f'<h2>Grokking search</h2><p class="note">A small patch-8 model on a small memorizable '
                 f'train set, constant LR, strong weight decay, plain CE. "grokked" = train≥0.95 then '
                 f'val≥0.60 with a ≥1000-step gap.</p>{gt}'
                 f'<figure>{_img("figures/grok_curves.png")}<figcaption>Train (blue) vs val (red) accuracy '
                 f'vs step (log scale); a delayed val rise after train saturates = grokking.</figcaption></figure>')

    body += '</div>'
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>MechInterpVision report</title><style>{REPORT_CSS}</style></head><body>{body}</body></html>')
    with open("report.html", "w") as f:
        f.write(html)
    log("wrote report.html")


def full_analysis(ckpt_path, device):
    model, ck = load_checkpoint(ckpt_path, device)
    bcfg = BlobConfig(**ck["blob_cfg"])
    ds = make_dataset(bcfg, 2500, seed=777)
    probes = A.run_probes(model, ds, device)
    A.plot_trajectory(probes)
    A.patch_embed_filters(model)
    routing = A.attention_routing(model, ds, device)
    A.attention_examples(model, ds, device)
    pos_base, pos_nopos = A.pos_ablation(model, ds, device)
    _, head_rows = A.head_ablation(model, ds, device)
    patching = A.activation_patching(model, ds, device)
    base_rate = routing[0]["blob_base_rate"] if routing else float("nan")
    return {"tag": os.path.basename(ckpt_path), "val_acc": ck.get("val_acc", float("nan")),
            "cls_acc": probes["cls_acc"], "logit_lens": probes["logit_lens"],
            "patch_presence_auc": probes["patch_presence_auc"], "patch_count_r2": probes["patch_count_r2"],
            "routing": routing, "base_rate": base_rate, "pos_base": pos_base, "pos_nopos": pos_nopos,
            "head_ablation": head_rows, "patching": patching}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_hours", type=float, default=6.0)
    ap.add_argument("--reserve_min", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--grok_steps", type=int, default=12000)
    ap.add_argument("--grok_wds", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--grok_sizes", type=int, nargs="+", default=[500, 1000])
    ap.add_argument("--grok_depth", type=int, default=2)
    ap.add_argument("--grok_width", type=int, default=64)
    ap.add_argument("--sweep_patches", type=int, nargs="+", default=[8])
    ap.add_argument("--sweep_depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--sweep_widths", type=int, nargs="+", default=[64, 128])
    args = ap.parse_args()
    device = pick_device("auto")
    deadline = START + args.budget_hours * 3600
    reserve = args.reserve_min * 60
    log(f"device={device}  budget={args.budget_hours}h  reserve={args.reserve_min}min")

    bcfg = BlobConfig()
    log("generating shared data ...")
    train = make_dataset(bcfg, args.n_train, seed=1)
    val = make_dataset(bcfg, 4000, seed=2)
    ana = make_dataset(bcfg, 1500, seed=123)
    data = (train, val, ana)
    results = load_results()
    groks = json.load(open(GROKJSON)) if os.path.exists(GROKJSON) else []
    if results or groks:
        log(f"resuming with {len(results)} sweep runs, {len(groks)} grok runs")

    def R(p, d, w, ep=args.epochs, nt=args.n_train, seed=0):
        run_one(p, d, w, ep, nt, seed, data, deadline, reserve, results)

    # 1) grokking phase-diagram FIRST (user priority -> guaranteed budget): wd x train-size
    log("=== stage 1: grokking search ===")
    for wd in args.grok_wds:
        for nt in args.grok_sizes:
            run_grok(nt, wd, 1e-3, args.grok_steps, data, deadline, reserve, groks,
                     depth=args.grok_depth, width=args.grok_width)
    # 2) architecture sweep grid (deadline guard trims to whatever fits)
    log("=== stage 2: architecture sweep ===")
    for p in args.sweep_patches:
        for w in args.sweep_widths:
            for d in args.sweep_depths:
                R(p, d, w)
    # 3) multi-seed the top-2 configs by combined score
    log("=== stage 3: multi-seed top-2 ===")
    for r in sorted(aggregate(results), key=lambda r: -r["combined"])[:2]:
        for seed in (1, 2):
            R(r["patch_size"], r["depth"], r["d_model"], seed=seed)

    # finalize: scoring, plots, full analysis on winner, report
    log("=== finalize ===")
    agg = aggregate(results)
    try:
        plot_results(agg)
        grok_plot(groks)
    except Exception:
        log(f"plot failed:\n{traceback.format_exc()}")
    full = None
    if agg:
        winner = max(agg, key=lambda r: r["combined"])
        log(f"winner by combined: {winner['tag']} (acc {winner['acc']:.3f}, clarity {winner['mechanism_clarity']:.3f})")
        try:
            if winner.get("best_ckpt"):
                full = full_analysis(winner["best_ckpt"], device)
        except Exception:
            log(f"full analysis failed:\n{traceback.format_exc()}")
    else:
        log("no sweep configs completed")
    write_report(agg, full, device, groks)
    log("ALL DONE")


if __name__ == "__main__":
    main()
