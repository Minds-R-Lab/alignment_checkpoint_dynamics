#!/usr/bin/env python
"""Exp 3 — is the aligned component load-bearing, and is it the radial contraction? (Gap 4)

Interventions on the write vectors of layers [--layers], evaluated on the fixed eval set:
  original | remove_aligned | remove_random (norm-matched, several seeds) | flip_aligned | keep_only_aligned
For each: loss, per-layer radial projection E[(mlp_out . h)/|h|^2], residual norms.
Also per-layer single-layer removal (which layer's component matters most).

  python scripts/03_ablation.py --model EleutherAI/pythia-70m --revision main --device cuda
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from rwloop.data import load_model, eval_batches
from rwloop.adapters import get_layers
from rwloop.hooks import collect
from rwloop.intervene import ablate, snapshot, restore
from rwloop.metrics import aligned_energy_fraction
from rwloop.io import outdir, save_json

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--revision", default="main")
p.add_argument("--layers", nargs="*", type=int, default=None, help="default: all but layer 0")
p.add_argument("--n_tokens", type=int, default=1_000_000); p.add_argument("--seq_len", type=int, default=2048)
p.add_argument("--batch_size", type=int, default=8); p.add_argument("--dataset", default="monology/pile-uncopyrighted")
p.add_argument("--dataset_config", default=None); p.add_argument("--n_random", type=int, default=3)
p.add_argument("--device", default="cuda")
args = p.parse_args()

model, tok = load_model(args.model, args.revision, device=args.device)
batches = eval_batches(tok, args.n_tokens, args.seq_len, args.batch_size, args.dataset, args.dataset_config)
layers = get_layers(model)
target = [layers[i] for i in (args.layers if args.layers is not None else range(1, len(layers)))]
snap = snapshot(layers)
out = {"model": args.model, "revision": args.revision, "layers": [L.idx for L in target],
       "aligned_energy": {L.idx: aligned_energy_fraction(L.read(), L.write()) for L in layers}, "runs": {}}

def run(tag):
    res, loss = collect(model, layers, batches, args.device, want_grad=False)
    out["runs"][tag] = dict(loss=loss, radial=res["radial"], resid_norm=res["resid_norm"])
    print(f"{tag:>24}: loss {loss:7.3f} | radial " + " ".join(f"{r:+.2f}" for r in res["radial"]) +
          " | norm " + " ".join(f"{r:6.1f}" for r in res["resid_norm"]), flush=True)
    restore(layers, snap)

run("original")
for mode in ["remove_aligned", "flip_aligned", "keep_only_aligned"]:
    ablate(target, mode); run(mode)
for s in range(args.n_random):
    ablate(target, "remove_random", torch.Generator().manual_seed(s)); run(f"remove_random_s{s}")
for L in target:  # single-layer removals
    ablate([L], "remove_aligned"); run(f"remove_aligned_L{L.idx}")
    ablate([L], "remove_random", torch.Generator().manual_seed(0)); run(f"remove_random_L{L.idx}")
save_json(out, os.path.join(outdir("03_ablation"), f"{args.model.replace('/', '_')}_{args.revision}.json"))
