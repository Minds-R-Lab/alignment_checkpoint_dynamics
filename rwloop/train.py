"""Continued pretraining from a checkpoint with per-unit snapshots.

Matched runs: same `seed` => same data order and same dropout/noise streams; interventions differ only
in the weight/bias manipulation applied at `intervene_at`. Snapshots record, at each step in
`snap_steps`: per-layer per-unit cos(read, write), and firing rates / gradient exposure averaged over
the `window` steps preceding the snapshot (so 'f at step s' means P(z>0) over steps s-window..s).
"""
from __future__ import annotations
import time
import numpy as np
import torch
from .adapters import get_layers, firing_module
from .metrics import unit_cos, ov_unit_cos
from .hooks import Recorder


def cos_snapshot(layers) -> dict:
    out = {"mlp": [unit_cos(L.read(), L.write()) for L in layers]}
    if layers[0].v_mod is not None:
        out["ov"] = [ov_unit_cos(L.v(), L.o()) for L in layers]
    return out


def continue_pretraining(model, stream, steps: int, lr: float, device, snap_steps: list[int], window: int = 30,
                         weight_decay: float = 0.1, grad_clip: float = 1.0, bf16: bool = True,
                         extra_params=(), on_step=None, log_every: int = 100, want_exposure: bool = True):
    """Train `steps` steps on `stream`. `on_step(step, model, opt)` is called AFTER the optimizer step
    (use it for interventions). Returns dict of snapshots keyed by step."""
    layers = get_layers(model)
    params = [p for p in model.parameters() if p.requires_grad] + list(extra_params)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    snaps, losses = {}, []
    rec = None
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        # start recording `window` steps before each snapshot
        if rec is None and any(s - window < step <= s for s in snap_steps):
            rec = Recorder(layers, want_grad=want_exposure).attach()
        x = next(stream).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16, enabled=bf16):
            out = model(input_ids=x, labels=x)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        if rec is not None:
            rec.count(x.numel())
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        losses.append(out.loss.item())
        if step in snap_steps:
            snap = cos_snapshot(layers)
            if rec is not None:
                snap.update(rec.results()); rec.detach(); rec = None
            snap["loss"] = float(np.mean(losses[-window:]))
            snaps[step] = snap
        if on_step is not None:
            on_step(step, model, opt)
        if step % log_every == 0:
            print(f"step {step:6d} loss {np.mean(losses[-log_every:]):.4f} "
                  f"L1 cos {np.mean(unit_cos(layers[min(1,len(layers)-1)].read(), layers[min(1,len(layers)-1)].write())):+.3f} [{time.time()-t0:.0f}s]", flush=True)
    model.eval()
    return snaps, losses
