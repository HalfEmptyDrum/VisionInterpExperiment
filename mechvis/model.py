"""A tiny hooked ViT for object counting.

Small enough to fully reverse-engineer (depth 3, 4 heads, d=128 by default),
pre-LN, learned positional embeddings, CLS-token classification head over counts
0..max_count.

Instrumentation
---------------
``run_with_cache(x) -> (logits, cache)`` returns a dict of detached intermediate
activations keyed by name:
  patch_embed, embed,
  blocks.{i}.resid_pre, blocks.{i}.attn.pattern (B,H,T,T), blocks.{i}.attn.z (B,T,H,dh),
  blocks.{i}.attn_out, blocks.{i}.resid_mid, blocks.{i}.mlp_post_act, blocks.{i}.mlp_out,
  blocks.{i}.resid_post, resid_final, cls_final, logits.

``run_with_hooks(x, hooks={name: fn})`` runs the same forward but passes each
named activation through ``fn(tensor) -> tensor`` *before* it flows downstream --
this is the single mechanism for both ablation (e.g. zero a head's z slice) and
activation patching (overwrite resid positions from a source run). ``ablate_pos``
drops the positional embedding.

``head_results(cache, layer)`` decomposes an attention layer's output into
per-head contributions to the residual stream (post W_O, bias excluded).

Sequence layout: position 0 is CLS; positions 1..n_patches are patch tokens in
row-major order (see data.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class ViTConfig:
    img_size: int = 64
    patch_size: int = 8
    in_ch: int = 1
    d_model: int = 128
    depth: int = 3
    n_heads: int = 4
    mlp_ratio: int = 4
    n_classes: int = 9

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


def _apply(name: str, t: torch.Tensor, cache: Optional[dict], hooks: Optional[dict]) -> torch.Tensor:
    """Pass ``t`` through ``hooks[name]`` if present, then record it in ``cache``.

    Returns the (possibly modified) tensor so it flows downstream -- enabling both
    observation (cache) and intervention (hooks) at the same point.
    """
    if hooks is not None and name in hooks:
        t = hooks[name](t)
    if cache is not None:
        cache[name] = t.detach()
    return t


class Attention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.scale = cfg.d_head**-0.5

    def forward(self, x, cache=None, hooks=None, prefix=""):
        B, T, D = x.shape
        H, dh = self.cfg.n_heads, self.cfg.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, dh)
        if cache is None and hooks is None:
            # fast path: fused attention (flash on CUDA); never materializes the
            # B x H x T x T probability tensor. Numerically equal to the explicit
            # path below, so training/eval results are unchanged.
            z = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)  # (B, T, H, dh)
            return self.proj(z.reshape(B, T, D))
        # explicit path: exposes / hooks the attention pattern and z for interpretability
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        pattern = _apply(prefix + "pattern", attn.softmax(dim=-1), cache, hooks)
        z = _apply(prefix + "z", (pattern @ v).transpose(1, 2), cache, hooks)  # (B, T, H, dh)
        return self.proj(z.reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, cfg.d_model)

    def forward(self, x, cache=None, hooks=None, prefix=""):
        x = _apply(prefix + "resid_pre", x, cache, hooks)
        attn_out = self.attn(self.ln1(x), cache, hooks, prefix + "attn.")
        attn_out = _apply(prefix + "attn_out", attn_out, cache, hooks)
        x = _apply(prefix + "resid_mid", x + attn_out, cache, hooks)
        act = _apply(prefix + "mlp_post_act", self.act(self.fc1(self.ln2(x))), cache, hooks)
        mlp_out = _apply(prefix + "mlp_out", self.fc2(act), cache, hooks)
        x = _apply(prefix + "resid_post", x + mlp_out, cache, hooks)
        return x


class HookedViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(
            cfg.in_ch, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches + 1, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        self._init()

    def _init(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cache: Optional[dict] = None, hooks: Optional[dict] = None,
                ablate_pos: bool = False):
        B = x.shape[0]
        p = rearrange(self.patch(x), "b d gh gw -> b (gh gw) d")
        p = _apply("patch_embed", p, cache, hooks)
        x = torch.cat([self.cls.expand(B, -1, -1), p], dim=1)
        if not ablate_pos:
            x = x + self.pos
        x = _apply("embed", x, cache, hooks)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache, hooks, f"blocks.{i}.")
        x = _apply("resid_final", self.ln_f(x), cache, hooks)
        cls_final = _apply("cls_final", x[:, 0], cache, hooks)
        return _apply("logits", self.head(cls_final), cache, hooks)

    @torch.no_grad()
    def run_with_cache(self, x) -> tuple[torch.Tensor, dict]:
        cache: dict = {}
        logits = self.forward(x, cache=cache)
        return logits, cache

    @torch.no_grad()
    def run_with_hooks(self, x, hooks: Optional[dict] = None, ablate_pos: bool = False,
                       return_cache: bool = False):
        cache: dict = {} if return_cache else None
        logits = self.forward(x, cache=cache, hooks=hooks, ablate_pos=ablate_pos)
        return (logits, cache) if return_cache else logits

    def head_results(self, cache: dict, layer: int) -> torch.Tensor:
        """Per-head contribution to the residual stream after W_O (bias excluded).

        Returns (B, T, H, d_model): result[..., h, :] is what head h writes.
        Summing over h (+ proj.bias) reconstructs ``blocks.{layer}.attn_out``.
        """
        z = cache[f"blocks.{layer}.attn.z"]  # (B, T, H, dh)
        B, T, H, dh = z.shape
        W_O = self.blocks[layer].attn.proj.weight.view(self.cfg.d_model, H, dh)
        return torch.einsum("bthd,ohd->btho", z, W_O.to(z.device))


def load_checkpoint(path: str, device: str = "cpu") -> tuple["HookedViT", dict]:
    """Rebuild a trained model from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HookedViT(ViTConfig(**ckpt["vit_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
