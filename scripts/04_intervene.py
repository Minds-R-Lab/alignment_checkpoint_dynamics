#!/usr/bin/env python
"""Exp 4/5 — PREREGISTERED causal tests by continued pretraining from a checkpoint (see PREREGISTRATION.md).

Arms (each arm is a full matched run; --seed fixes the data order for the whole arm set):
  control                      no intervention
  component:<alpha>            d_k <- d_k + alpha * proj_k on layers L1..  (alpha=-1 remove, +2 triple)   [c -> f]
  bias:<beta>:<select>         +/-beta*std virtual bias on two random (or aligned-selected) 20% groups    [f -> c]
Snapshots (cos per unit, firing over the preceding window, gradient exposure) at --snap_steps after
the intervention step. Each arm writes results/04_intervene/<tag>_seed<k>.pkl; 05_analyze.py compares.

Typical H100 plan for Pythia-70m at step 8000 (~1.5 min per 1000 steps at 32x2048 tokens/step):
  for s in 0 1 2 3 4; do
    python scripts/04_intervene.py --model EleutherAI/pythia-70m --revision step8000 --arm control --seed $s
    python scripts/04_intervene.py ... --arm component:-1 --seed $s
    python scripts/04_intervene.py ... --arm component:2  --seed $s
    for b in 0.5 1 2; do python scripts/04_intervene.py ... --arm bias:$b:random --seed $s; done
  done
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from rwloop.data import load_model, token_stream, eval_batches
from rwloop.adapters import get_layers
from rwloop.train import continue_pretraining, cos_snapshot
from rwloop.hooks import collect, eval_loss
from rwloop.intervene import scale_aligned, shift_bias
from rwloop.io import outdir, save

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--revision", default="step8000")
p.add_argument("--arm", required=True, help="control | component:<alpha> | bias:<beta>:<random|aligned>")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--layers", nargs="*", type=int, default=None, help="intervened layers; default all but 0")
p.add_argument("--steps", type=int, default=8000); p.add_argument("--intervene_at", type=int, default=200,
               help="steps of warm-up training BEFORE the intervention (lets Adam state settle)")
p.add_argument("--snap_after", nargs="+", type=int, default=[0, 30, 500, 2000, 8000],
               help="snapshot at intervene_at + these offsets (0 = immediately before intervention)")
p.add_argument("--window", type=int, default=30)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--seq_len", type=int, default=2048); p.add_argument("--bias_frac", type=float, default=0.2)
p.add_argument("--dataset", default="monology/pile-uncopyrighted"); p.add_argument("--dataset_config", default=None)
p.add_argument("--device", default="cuda"); p.add_argument("--bf16", action="store_true")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
model, tok = load_model(args.model, args.revision, device=args.device)
layers = get_layers(model)
target = [layers[i] for i in (args.layers if args.layers is not None else range(1, len(layers)))]
stream = token_stream(tok, args.dataset, args.dataset_config, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed)
evalb = eval_batches(tok, 500_000, args.seq_len, 8, args.dataset, args.dataset_config)
snap_steps = sorted({args.intervene_at + d for d in args.snap_after} | {args.steps})
total = max(snap_steps)

state = {"groups": None, "vbs": [], "cos_pre": None}
first_batch = next(token_stream(tok, args.dataset, args.dataset_config, seq_len=args.seq_len, batch_size=4, seed=999))

def on_step(step, model, opt):
    if step != args.intervene_at:
        return
    state["cos_pre"] = cos_snapshot(layers)
    kind = args.arm.split(":")
    if kind[0] == "control":
        return
    if kind[0] == "component":
        scale_aligned(target, float(kind[1]))
    elif kind[0] == "bias":
        beta, select = float(kind[1]), (kind[2] if len(kind) > 2 else "random")
        cos_by_layer = {L.idx: state["cos_pre"]["mlp"][L.idx] for L in layers}
        vbs, groups = shift_bias(model, target, beta, args.bias_frac, args.seed * 1000 + 123, first_batch, args.device, select, cos_by_layer)
        state["vbs"], state["groups"] = vbs, groups
        # register the new trainable biases with the optimizer of the running loop
        opt.add_param_group({"params": [vb.b for vb in vbs], "weight_decay": 0.0})
    print(f"[arm {args.arm}] intervened at step {step}; L1 cos now {np.mean(state['cos_pre']['mlp'][min(1, len(layers)-1)]):+.3f}", flush=True)

snaps, losses = continue_pretraining(model, stream, total, args.lr, args.device, snap_steps, window=args.window,
                                     bf16=args.bf16, on_step=on_step, want_exposure=True)
for vb in state["vbs"]:
    vb.detach()
final_eval = eval_loss(model, evalb, args.device)
tag = args.arm.replace(":", "_")
out = dict(model=args.model, revision=args.revision, arm=args.arm, seed=args.seed, intervene_at=args.intervene_at,
           layers=[L.idx for L in target], snaps=snaps, losses=losses, groups=state["groups"], cos_pre=state["cos_pre"],
           bias_final=[vb.b.detach().cpu().numpy() for vb in state["vbs"]], eval_loss=final_eval)
save(out, os.path.join(outdir("04_intervene"), f"{args.model.replace('/', '_')}_{args.revision}_{tag}_seed{args.seed}.pkl"))
print(f"done {tag} seed {args.seed}: eval loss {final_eval:.4f}")
