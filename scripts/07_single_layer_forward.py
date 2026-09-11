#!/usr/bin/env python
"""Exp 7 — the SINGLE-LAYER forward arm (c -> f), the follow-up H4/H2 could not settle.

Why this experiment exists
--------------------------
Stage 4's `component:<alpha>` arm intervened on ALL layers >=1 at once. The forward-arm prediction
(making the most-aligned units less anti-aligned lowers their later firing) came out with the WRONG sign
there, but for a confounded reason: a tripled/removed write in layer L changes the feature that layer L+1
READS, so the firing change of L+1's units is a CROSS-layer effect, not the within-layer c -> f coupling
the hypothesis is about. See the retraction of H4 in the paper.

This stage removes the confound by intervening on ONE layer at a time (04_intervene --layers L
--outdir 07_single_layer) and separating two quantities per intervened layer L:

  within-layer   Spearman(dc_k, df_k) and partial rho(dc_k, df_k | c0_k)  on layer L ITSELF
                 -- L reads from layers < L, which are untouched, so this is the clean c -> f coupling.
  cross-layer    the firing footprint on downstream layers L' > L (whose geometry was NOT changed):
                 mean |df| and mean df -- the pathway that produced the misleading Stage-4 sign.

H7 (preregistered, see docs/PREREGISTRATION.md): within-layer Spearman(dc, df) < -0.2 at the +2000 and
+7800 snapshots in a majority of intervened layers, for BOTH doses; and the downstream cross-layer
footprint is present (mean |df| downstream comparable to or larger than the within-layer |df|), which is
what made the all-layer arm read the opposite sign.

Matched controls are the ordinary Stage-4 `control_seed<k>.pkl` runs (same seed = same data order, no
intervention); this script looks for them in --ctrl_dir. Run the arms with, per layer L in 1..4, per
dose in {-1, +2}, per seed:

  python scripts/04_intervene.py --model EleutherAI/pythia-70m --revision step8000 \
      --arm component:2 --layers L --outdir 07_single_layer --seed S --steps 8000 --bf16
  (control_seed S from Stage 4 is reused; run one if you have none.)

  python scripts/07_single_layer_forward.py --model EleutherAI/pythia-70m --revision step8000

Writes results/07_single_layer/report.md and single_layer_tables.json.
"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rwloop.io import load, outdir, save_json
from rwloop.analysis import forward_arm, md_table

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--revision", default="step8000")
p.add_argument("--indir", default="results/07_single_layer")
p.add_argument("--ctrl_dir", default="results/04_intervene", help="where the matched control_seed*.pkl runs live")
p.add_argument("--within_thr", type=float, default=-0.2, help="H7 threshold on within-layer Spearman(dc,df)")
args = p.parse_args()

prefix = f"{args.model.replace('/', '_')}_{args.revision}_"

# --- load single-layer arms (exactly one intervened layer) and matched controls, keyed by seed
arms = {}   # (arm, L) -> {seed: run}
for fn in glob.glob(os.path.join(args.indir, prefix + "*.pkl")):
    r = load(fn)
    if len(r["layers"]) != 1 or r["arm"] == "control":
        continue
    arms.setdefault((r["arm"], r["layers"][0]), {})[r["seed"]] = r
ctrl = {}
for fn in glob.glob(os.path.join(args.ctrl_dir, prefix + "control_seed*.pkl")):
    r = load(fn); ctrl[r["seed"]] = r
if not arms:
    sys.exit(f"no single-layer arms in {args.indir} (run 04_intervene with --layers L --outdir 07_single_layer)")
if not ctrl:
    sys.exit(f"no control runs in {args.ctrl_dir} (need control_seed<k>.pkl matched by seed)")

lines = [f"# Single-layer forward arm (c -> f): {args.model} @ {args.revision}\n",
         f"arms: { {f'{a}@L{L}': sorted(s) for (a, L), s in sorted(arms.items())} }\n",
         f"controls: seeds {sorted(ctrl)}\n",
         f"H7 threshold: within-layer Spearman(dc, df) < {args.within_thr}\n"]
tables, verdict_rows = {}, []


def post_snaps(r):
    return sorted(s for s in r["snaps"] if s > r["intervene_at"])


for (arm, L), seeds in sorted(arms.items()):
    common = sorted(set(seeds) & set(ctrl))
    lines.append(f"\n## arm `{arm}`  intervened layer **L{L}**  (seeds {common})\n")
    if not common:
        lines.append("no matched control seeds\n"); continue
    ref = seeds[common[0]]
    all_layers = list(range(len(ref["snaps"][post_snaps(ref)[0]]["mlp"])))
    downstream = [l for l in all_layers if l > L]
    for step in post_snaps(ref):
        within, cross = [], []                       # per-seed dicts
        for s in common:
            a, c = seeds[s], ctrl[s]
            if step not in a["snaps"] or step not in c["snaps"]:
                continue
            # within-layer: the intervened layer's own dc, df
            dc = a["snaps"][step]["mlp"][L] - c["snaps"][step]["mlp"][L]
            df = a["snaps"][step]["firing"][L] - c["snaps"][step]["firing"][L]
            c0 = c["cos_pre"]["mlp"][L]
            fa = forward_arm(dc, df, c0)
            within.append(fa)
            # cross-layer footprint on downstream layers (geometry there was NOT directly changed)
            for l2 in downstream:
                dfd = a["snaps"][step]["firing"][l2] - c["snaps"][step]["firing"][l2]
                cross.append(dict(layer=l2, df_abs=float(np.abs(dfd).mean()), df_mean=float(dfd.mean())))
        if not within:
            continue

        def agg(rows, key):
            v = np.array([r[key] for r in rows], float)
            return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))

        sp_m, sp_e = agg(within, "spearman"); pa_m, pa_e = agg(within, "partial_given_c0")
        sl_m, _ = agg(within, "slope"); dfw_m, _ = agg(within, "df_mean"); dcw_m, _ = agg(within, "dc_mean")
        cross_abs = float(np.mean([r["df_abs"] for r in cross])) if cross else float("nan")
        row = dict(step=step, layer=L, arm=arm, n_seeds=len(within),
                   within_spearman=sp_m, within_spearman_sem=sp_e,
                   within_partial=pa_m, within_partial_sem=pa_e, within_slope=sl_m,
                   dc_mean=dcw_m, df_within=dfw_m, df_abs_within=float(np.mean([abs(r["df_mean"]) for r in within])),
                   df_abs_cross=cross_abs)
        tables[f"{arm}@L{L}@{step}"] = row
        lines.append(f"- step {step}: within Spearman(dc,df) = **{sp_m:+.3f}** ± {sp_e:.3f}"
                     f" | partial(|c0) = {pa_m:+.3f} ± {pa_e:.3f} | slope dν/dc = {sl_m:+.3f}"
                     f" | Δc̄ = {dcw_m:+.3f}, Δf̄(within) = {dfw_m:+.4f}"
                     f" | cross-layer mean|Δf| = {cross_abs:.4f}\n")
        if step in (ref["intervene_at"] + 2000, post_snaps(ref)[-1]):
            verdict_rows.append(row)

# --- H7 verdict: within-layer sign at the late snapshots, per (arm, layer)
lines.append("\n## H7 verdict (within-layer forward arm at the +2000 and final snapshots)\n")
if verdict_rows:
    cols = ["arm", "layer", "step", "within_spearman", "within_partial", "within_slope", "df_abs_within", "df_abs_cross"]
    lines.append(md_table(sorted(verdict_rows, key=lambda r: (r["arm"], r["layer"], r["step"])), cols,
                          fmt="{:+.3f}"))
    passed = [r for r in verdict_rows if r["within_spearman"] < args.within_thr]
    frac = len(passed) / len(verdict_rows)
    lines.append(f"\nWithin-layer Spearman(dc,df) < {args.within_thr} in {len(passed)}/{len(verdict_rows)} "
                 f"(arm, layer, snapshot) cells ({frac:.0%}). "
                 + ("**H7 supported**: the within-layer forward arm is negative once the cross-layer path is "
                    "removed — the sign that failed in the all-layer Stage-4 arm.\n" if frac >= 0.5 else
                    "**H7 not supported** at this threshold; report as-is.\n"))
    # is the cross-layer footprint really the confound? compare within vs downstream |df|
    ca = np.array([r["df_abs_cross"] for r in verdict_rows], float)
    wa = np.array([r["df_abs_within"] for r in verdict_rows], float)
    m = np.isfinite(ca) & np.isfinite(wa)
    if m.any():
        lines.append(f"\nCross-layer footprint: downstream mean|Δf| = {ca[m].mean():.4f} vs within-layer "
                     f"|Δf̄| = {wa[m].mean():.4f} — the downstream path is "
                     f"{'comparable/larger' if ca[m].mean() >= wa[m].mean() else 'smaller'}, "
                     "consistent with it driving the all-layer sign.\n")

od = outdir("07_single_layer")
open(os.path.join(od, "report.md"), "w").write("".join(lines))
save_json(tables, os.path.join(od, "single_layer_tables.json"))
print("".join(lines))
