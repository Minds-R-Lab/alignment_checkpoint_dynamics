#!/usr/bin/env python
"""Exp 2 — activation-space observables at chosen checkpoints, on in-distribution text (Gap 1/2 mechanism).

Per (model, revision): per-unit firing rate P(z>0), gradient exposure E[h_k (g . a_hat_k)], per-layer
radial projection and residual norms, eval loss. Then, across revisions: unit-trajectory stability,
cross-lagged partial correlations (geometry -> firing vs firing -> geometry), monotone decile profile.

  python scripts/02_activations.py --model EleutherAI/pythia-70m --revisions step8000 step32000 step143000 \
      --n_tokens 2000000 --device cuda
Output: results/02_activations/<model>_<rev>.pkl, cross_lagged.json, trajectories.json
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scipy.stats import spearmanr
from rwloop.data import load_model, eval_batches
from rwloop.adapters import get_layers
from rwloop.hooks import collect
from rwloop.metrics import unit_cos
from rwloop.analysis import cross_lagged, trajectory_stability
from rwloop.io import outdir, save, load, save_json

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--revisions", nargs="+", required=True)
p.add_argument("--n_tokens", type=int, default=2_000_000)
p.add_argument("--seq_len", type=int, default=2048)
p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--dataset", default="monology/pile-uncopyrighted")
p.add_argument("--dataset_config", default=None)
p.add_argument("--device", default="cuda")
p.add_argument("--no_grad", action="store_true", help="skip gradient exposure (faster)")
args = p.parse_args()

od = outdir("02_activations")
per_rev = {}
for rev in args.revisions:
    fn = os.path.join(od, f"{args.model.replace('/', '_')}_{rev}.pkl")
    if os.path.exists(fn):
        per_rev[rev] = load(fn); print("loaded", fn); continue
    model, tok = load_model(args.model, rev, device=args.device)
    batches = eval_batches(tok, args.n_tokens, args.seq_len, args.batch_size, args.dataset, args.dataset_config)
    layers = get_layers(model)
    res, loss = collect(model, layers, batches, args.device, want_grad=not args.no_grad)
    res["cos"] = [unit_cos(L.read(), L.write()) for L in layers]
    res["loss"] = loss; res["rev"] = rev
    save(res, fn); per_rev[rev] = res
    print(f"{rev}: loss {loss:.3f} | radial " + " ".join(f"{r:+.2f}" for r in res["radial"]) +
          " | resid norm " + " ".join(f"{r:.1f}" for r in res["resid_norm"]), flush=True)
    for L in layers:
        c, f = res["cos"][L.idx], res["firing"][L.idx]
        line = f"  L{L.idx:2d} rho(f,cos)={spearmanr(f, c)[0]:+.2f}"
        if res["exposure"][L.idx] is not None:
            line += f"  rho(exposure,cos)={spearmanr(res['exposure'][L.idx], c)[0]:+.2f}"
        print(line)
    del model; torch.cuda.empty_cache()

revs = args.revisions
if len(revs) >= 2:
    e, l = per_rev[revs[0]], per_rev[revs[-1]]
    cl, tr = {}, {}
    for i in range(len(e["cos"])):
        cl[i] = cross_lagged(e["cos"][i], e["firing"][i], l["cos"][i], l["firing"][i])
        tr[i] = trajectory_stability(e["cos"][i], l["cos"][i])
        print(f"L{i}: geom->fire {cl[i]['geom_to_fire']:+.2f}  fire->geom {cl[i]['fire_to_geom']:+.2f}  "
              f"corr(c_early,c_late) {tr[i]['corr']:.2f}  tail-from-leaders {tr[i]['tail_from_leaders']:.2f}")
    save_json({"early": revs[0], "late": revs[-1], "cross_lagged": cl}, os.path.join(od, "cross_lagged.json"))
    save_json({"early": revs[0], "late": revs[-1], "trajectories": tr}, os.path.join(od, "trajectories.json"))
