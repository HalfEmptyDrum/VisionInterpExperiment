"""Quick end-to-end sanity check: data, baseline, and a forward pass with cache.

    python -m mechvis.sanity
"""
from __future__ import annotations

from .data import BlobConfig, intensity_baseline, make_dataset
from .model import HookedViT, ViTConfig
from .viz import montage


def main():
    bcfg = BlobConfig()
    ds = make_dataset(bcfg, 2000, seed=0)

    path = montage(ds["images"], ds["counts"], "figures/sanity_blobs.png")
    print(f"wrote {path}")

    base = intensity_baseline(ds["images"], ds["counts"])
    print(
        "intensity baseline (lower=harder): "
        f"r={base['pearson_r']:.3f} R^2={base['r2']:.3f} "
        f"round_acc={base['lin_round_acc']:.3f}"
    )

    pc_total = ds["patch_counts"].sum(1)
    print("patch_counts sum == count for all images:", bool((pc_total == ds["counts"]).all()))

    # class balance
    binc = ds["counts"].bincount(minlength=bcfg.n_classes).tolist()
    print("class histogram:", binc)

    mcfg = ViTConfig(
        img_size=bcfg.img_size, patch_size=bcfg.patch_size, in_ch=1, n_classes=bcfg.n_classes
    )
    model = HookedViT(mcfg).eval()
    logits, cache = model.run_with_cache(ds["images"][:8])
    print(f"\nlogits {tuple(logits.shape)}  params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")
    for k, v in cache.items():
        print(f"  {k:28s} {tuple(v.shape)}")

    # per-head attribution reconstruction check
    hr = model.head_results(cache, layer=0).sum(2) + model.blocks[0].attn.proj.bias
    err = (hr - cache["blocks.0.attn_out"]).abs().max().item()
    print(f"\nhead_results reconstruction max-abs-err: {err:.2e} (should be ~0)")


if __name__ == "__main__":
    main()
