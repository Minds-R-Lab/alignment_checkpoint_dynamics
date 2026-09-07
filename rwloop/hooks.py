"""Activation- and gradient-space observables collected with hooks.

IMPORTANT: every hook here returns None. A forward hook that returns a tensor REPLACES the module
output (this bug silently destroyed a run during development); tests/test_hooks.py guards it.

Per layer we collect, over a stream of token batches:
  firing[k]     = P(pre-activation_k > 0)               (GeLU: up output; gated: gate output)
  exposure[k]   = E_t[ h_{k,t} * (g_t . a_hat_k) ]      gradient exposure of unit k along its read
                  direction, where h = post-activation input to write_mod, g = dL/d(mlp_out)
  radial[L]     = E_t[ (mlp_out_t . h_t) / ||h_t||^2 ]  projection of the MLP write onto the residual
  resid_norm[L] = E_t ||h_t||                            residual norm entering the MLP norm
"""
from __future__ import annotations
import numpy as np
import torch
from .adapters import LayerHandles, firing_module


class Recorder:
    def __init__(self, layers: list[LayerHandles], want_grad: bool = False):
        self.layers = layers
        self.want_grad = want_grad
        self._h = {}
        self.reset()
        self.handles = []

    def reset(self):
        n = len(self.layers)
        self.n_tok = 0
        self.fire = [None] * n
        self.expo = [None] * n
        self.radial = [0.0] * n
        self.rnorm = [0.0] * n
        self._hpost = [None] * n        # post-activation h (T, m) for exposure
        self._resid = [None] * n        # residual h (T, d)

    def _add(self, arr, i, val):
        arr[i] = val if arr[i] is None else arr[i] + val

    def attach(self):
        for L in self.layers:
            i = L.idx
            fm = firing_module(L)

            def pre_hook(mod, inp, out, i=i):
                o = out.detach()
                self._add(self.fire, i, (o > 0).float().sum(dim=(0, 1)).cpu())
                return None

            def resid_hook(mod, args, i=i):
                h = args[0].detach()
                self._resid[i] = h.reshape(-1, h.shape[-1])
                self.rnorm[i] += h.norm(dim=-1).sum().item()
                return None

            def write_in_hook(mod, args, i=i):
                if self.want_grad:
                    self._hpost[i] = args[0].detach().reshape(-1, args[0].shape[-1])
                return None

            def mlp_out_hook(mod, inp, out, i=i):
                o = out.detach().reshape(-1, out.shape[-1]); h = self._resid[i]
                self.radial[i] += ((o * h).sum(-1) / (h * h).sum(-1).clamp_min(1e-12)).sum().item()
                return None

            def grad_hook(mod, grad_in, grad_out, i=i, L=L):
                g = grad_out[0].detach().reshape(-1, grad_out[0].shape[-1]).float()   # (T, d)
                hpost = self._hpost[i].float()                                         # (T, m)
                A = L.read(); Ah = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)   # (m, d)
                expo = ((hpost.T @ g) * Ah.to(g.device)).sum(1)                        # (m,)
                self._add(self.expo, i, expo.cpu())
                return None

            self.handles += [fm.register_forward_hook(pre_hook),
                             L.mlp_norm.register_forward_pre_hook(resid_hook),
                             L.write_mod.register_forward_pre_hook(write_in_hook),
                             L.mlp.register_forward_hook(mlp_out_hook)]
            if self.want_grad:
                self.handles.append(L.mlp.register_full_backward_hook(grad_hook))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def count(self, n_tokens: int):
        self.n_tok += n_tokens

    def results(self) -> dict:
        n = max(self.n_tok, 1)
        return dict(
            firing=[None if f is None else (f / n).numpy() for f in self.fire],
            exposure=[None if e is None else (e / n).numpy() for e in self.expo],
            radial=[r / n for r in self.radial],
            resid_norm=[r / n for r in self.rnorm])


@torch.no_grad()
def eval_loss(model, batches, device) -> float:
    tot, n = 0.0, 0
    for x in batches:
        x = x.to(device)
        out = model(input_ids=x, labels=x)
        tot += out.loss.item() * x.numel(); n += x.numel()
    return tot / n


def collect(model, layers, batches, device, want_grad=False) -> tuple[dict, float]:
    """Run batches through the model with a Recorder attached. If want_grad, runs backward to get
    dL/d(mlp_out) (no optimizer step). Returns (recorder results, mean loss)."""
    rec = Recorder(layers, want_grad=want_grad).attach()
    tot, n = 0.0, 0
    try:
        for x in batches:
            x = x.to(device)
            if want_grad:
                model.zero_grad(set_to_none=True)
                out = model(input_ids=x, labels=x)
                out.loss.backward()
            else:
                with torch.no_grad():
                    out = model(input_ids=x, labels=x)
            rec.count(x.numel()); tot += out.loss.item() * x.numel(); n += x.numel()
    finally:
        rec.detach()
        model.zero_grad(set_to_none=True)
    return rec.results(), tot / n
