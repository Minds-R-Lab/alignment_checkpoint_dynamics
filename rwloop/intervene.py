"""Manipulations of the per-unit aligned component and of firing rates.

Decomposition per unit k:  d_k = (d_k . a_hat_k) a_hat_k + r_k   (aligned + orthogonal residual)

Ablations (evaluation only; use `restore` afterwards):
  remove_aligned        d_k <- r_k
  flip_aligned          d_k <- d_k - 2 proj      (sign of aligned component reversed)
  keep_only_aligned     d_k <- proj
  remove_random         d_k <- d_k - ||proj|| * u_k, u_k random unit vector ⟂ a_hat_k   (norm-matched control)

Interventions (continued training):
  scale_aligned(alpha)  d_k <- d_k + alpha * proj      (alpha=-1 removes, alpha=+2 triples)
  VirtualBias           adds a trainable per-unit bias to the firing pre-activation (works for
                        gated models that have no bias); initialized to +/- beta*std for random groups
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from .adapters import LayerHandles, firing_module


def _proj(L: LayerHandles):
    A = L.read_mod.weight.detach().float()
    Ah = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)      # (m, d)
    D = L.write_mod.weight.detach().float().T                    # (m, d) rows = write vectors
    proj = (D * Ah).sum(1, keepdim=True) * Ah
    return D, Ah, proj


def snapshot(layers: list[LayerHandles]) -> dict:
    return {L.idx: L.write_mod.weight.detach().clone() for L in layers}


def restore(layers: list[LayerHandles], snap: dict):
    with torch.no_grad():
        for L in layers:
            L.write_mod.weight.copy_(snap[L.idx])


@torch.no_grad()
def ablate(layers: list[LayerHandles], mode: str, rng: torch.Generator | None = None):
    for L in layers:
        D, Ah, proj = _proj(L)
        if mode == "original":
            Dn = D
        elif mode == "remove_aligned":
            Dn = D - proj
        elif mode == "flip_aligned":
            Dn = D - 2 * proj
        elif mode == "keep_only_aligned":
            Dn = proj
        elif mode == "remove_random":
            R = torch.randn(D.shape, generator=rng, device="cpu").to(D.device)
            R = R - (R * Ah).sum(1, keepdim=True) * Ah
            R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Dn = D - proj.norm(dim=1, keepdim=True) * R
        else:
            raise ValueError(mode)
        L.write_mod.weight.copy_(Dn.T.to(L.write_mod.weight.dtype))


@torch.no_grad()
def scale_aligned(layers: list[LayerHandles], alpha: float):
    for L in layers:
        D, Ah, proj = _proj(L)
        L.write_mod.weight.copy_((D + alpha * proj).T.to(L.write_mod.weight.dtype))


class VirtualBias(nn.Module):
    """Trainable additive bias on a module's output, applied by wrapping the module's forward.
    Wrapping (rather than a forward hook) guarantees that EVERY hook registered on the module -
    including Recorder's firing-rate hook - sees the shifted pre-activation regardless of registration
    order. `detach()` restores the original forward."""
    def __init__(self, n: int, device):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(n, device=device))
        self.mod = None; self._orig = None

    def attach(self, mod: nn.Module):
        self.mod, self._orig = mod, mod.forward
        b = self.b
        def fwd(x, *a, **k):
            out = self._orig(x, *a, **k)
            return out + b.to(out.dtype)
        mod.forward = fwd
        return self

    def detach(self):
        if self.mod is not None:
            self.mod.forward = self._orig; self.mod = None


@torch.no_grad()
def _prestd(layers, model, batch, device) -> dict[int, float]:
    """std of firing pre-activations per layer on one batch (sets the bias scale)."""
    stats, hs = {}, []
    for L in layers:
        def hk(m, i, o, L=L):
            stats[L.idx] = o.detach().float().std().item()
            return None
        hs.append(firing_module(L).register_forward_hook(hk))
    model(input_ids=batch.to(device))
    for h in hs:
        h.remove()
    return stats


def shift_bias(model, layers: list[LayerHandles], beta_std: float, frac: float, seed: int, batch, device,
               select: str = "random", cos_by_layer: dict | None = None):
    """Attach VirtualBias to each layer's firing module and initialize +beta*std on one group of units
    and -beta*std on another. Returns (list of VirtualBias, {layer: group array in {+1,0,-1}}).
    select='random' (preregistered default) or 'aligned' (top/bottom by cos; confounded, for comparison)."""
    stds = _prestd(layers, model, batch, device)
    g = torch.Generator().manual_seed(seed)
    vbs, groups = [], {}
    for L in layers:
        m = L.read_mod.weight.shape[0]
        k = int(frac * m)
        if select == "random":
            perm = torch.randperm(m, generator=g)
            up_idx, dn_idx = perm[:k], perm[k:2 * k]
        else:
            order = torch.argsort(torch.tensor(cos_by_layer[L.idx]))
            up_idx, dn_idx = order[:k], order[-k:]        # most anti-aligned get +, least get -
        vb = VirtualBias(m, device).attach(firing_module(L))
        with torch.no_grad():
            vb.b[up_idx] += beta_std * stds[L.idx]
            vb.b[dn_idx] -= beta_std * stds[L.idx]
        grp = np.zeros(m, dtype=int); grp[up_idx.numpy()] = 1; grp[dn_idx.numpy()] = -1
        vbs.append(vb); groups[L.idx] = grp
    return vbs, groups
