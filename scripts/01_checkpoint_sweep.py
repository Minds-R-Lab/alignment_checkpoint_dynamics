#!/usr/bin/env python
"""Exp 1 — read/write alignment across training checkpoints and model sizes (Gap 1 + Gap 3).

For each (model, revision): per-layer summary of cos(read, write), tail fractions, aligned energy,
block trace ratio, attention OV cos, and the permutation + rotation controls. Weight-only: fast.

Example (Pythia family, coarse checkpoint grid):
  python scripts/01_checkpoint_sweep.py --models EleutherAI/pythia-70m EleutherAI/pythia-160m \
      EleutherAI/pythia-410m EleutherAI/pythia-1b --checkpoints coarse
Gated family with published checkpoints (OLMo-2 1B; revisions like 'stage1-step10000-tokens21B'):
  python scripts/01_checkpoint_sweep.py --models allenai/OLMo-2-0425-1B --revisions main stage1-step10000-tokens21B
Output: results/01_checkpoint_sweep/<model>_<rev>.pkl and a summary CSV.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from rwloop.data import load_model, pythia_checkpoints
from rwloop.adapters import get_layers
from rwloop import metrics as M
from rwloop.io import outdir, save

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", required=True)
p.add_argument("--checkpoints", default=None, help="pythia grid: early|coarse|standard (ignored if --revisions)")
p.add_argument("--revisions", nargs="*", default=None)
p.add_argument("--device", default="cpu", help="weights only; cpu is fine")
p.add_argument("--n_controls", type=int, default=3)
args = p.parse_args()

od = outdir("01_checkpoint_sweep")
rng = np.random.default_rng(0)
rows = []
revs = args.revisions if args.revisions else (pythia_checkpoints(args.checkpoints) if args.checkpoints else ["main"])
for name in args.models:
    for rev in revs:
        try:
            model, _ = load_model(name, rev, device=args.device)
        except Exception as e:
            print(f"skip {name}@{rev}: {e}"); continue
        layers = get_layers(model)
        rec = {"model": name, "rev": rev, "layers": []}
        for L in layers:
            R, W = L.read(), L.write()
            s = M.summary(R, W)
            c = M.unit_cos(R, W)
            perm = np.mean([M.permutation_control(R, W, rng).mean() for _ in range(args.n_controls)])
            rot = [M.rotation_control(R, W, rng) for _ in range(args.n_controls)]
            s.update(perm_mean=float(perm), rot_mean=float(np.mean([r.mean() for r in rot])),
                     rot_std=float(np.mean([r.std() for r in rot])), n_units=len(c))
            if L.gate_mod is not None:
                G = L.gate()
                s.update(cos_write_gate_mean=float(M.unit_cos(G, W).mean()),
                         cos_write_gate_abs_gt02=float(np.mean(np.abs(M.unit_cos(G, W)) > 0.2)),
                         cos_up_gate_mean=float(M.unit_cos(G, R.T).mean()))   # both (m,d): pass R as (d,m)
            if L.v_mod is not None:
                ov = M.ov_unit_cos(L.v(), L.o())
                s.update(ov_mean=float(ov.mean()), ov_frac_gt05=float(np.mean(ov > 0.5)), ov_frac_lt05=float(np.mean(ov < -0.5)))
            rec["layers"].append({"idx": L.idx, "cos": c, **s})
            rows.append({"model": name, "rev": rev, "layer": L.idx, "rel_depth": L.idx / max(len(layers) - 1, 1), **{k: v for k, v in s.items()}})
            print(f"{name}@{rev} L{L.idx:2d}: mean {s['mean']:+.3f} frac<-0.5 {s['frac_lt']:.3f} energy {s['aligned_energy']:.3f} "
                  f"tr/F {s['trace_ratio']:+.2f} perm {perm:+.3f} rot {s['rot_mean']:+.3f}(std {s['rot_std']:.3f})" +
                  (f" ov {s['ov_mean']:+.3f}" if 'ov_mean' in s else ""), flush=True)
        save(rec, os.path.join(od, f"{name.replace('/', '_')}_{rev}.pkl"))
        del model; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(os.path.join(od, "summary.csv"), index=False)
print("wrote", os.path.join(od, "summary.csv"))
