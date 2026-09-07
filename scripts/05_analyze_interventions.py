#!/usr/bin/env python
"""Exp 5 — analysis of 04_intervene runs, matched by seed (arm vs control with identical data order).

Reports, per arm and snapshot offset, per layer:
  component arms:  Spearman(delta firing, cos_pre) and delta f for most/least aligned 20%  [c -> f]
  bias arms:       delta cos by group (+/0/-), per-unit Spearman(realized delta f, delta cos), dose-response
                   fits pooled over doses, growth law on controls; seed means +/- s.e.m.
  all arms:        eval loss, attractor regrowth (cos after remove vs control)
Writes results/05_analysis/report.md and tables.json.

  python scripts/05_analyze_interventions.py --model EleutherAI/pythia-70m --revision step8000
"""
import argparse, os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import spearmanr
from rwloop.io import load, outdir, save_json
from rwloop.analysis import dose_response, growth_law, md_table

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--revision", default="step8000")
p.add_argument("--indir", default="results/04_intervene")
args = p.parse_args()
prefix = f"{args.model.replace('/', '_')}_{args.revision}_"
runs = {}
for fn in glob.glob(os.path.join(args.indir, prefix + "*.pkl")):
    r = load(fn); runs.setdefault(r["arm"], {})[r["seed"]] = r
if "control" not in runs:
    sys.exit("no control runs found")
ctrl = runs["control"]
lines = [f"# Intervention analysis: {args.model} @ {args.revision}\n", f"arms: {sorted(runs)}; seeds per arm: { {a: sorted(v) for a, v in runs.items()} }\n"]
tables = {}

def snaps_after(r):
    """snapshot steps strictly after the intervention, sorted"""
    return sorted(s for s in r["snaps"] if s > r["intervene_at"])

for arm, seeds in runs.items():
    if arm == "control":
        continue
    kind = arm.split(":")[0]
    lines.append(f"\n## arm `{arm}`\n")
    common = sorted(set(seeds) & set(ctrl))
    if not common:
        lines.append("no matched control seeds\n"); continue
    for step in snaps_after(seeds[common[0]]):
        rows = []
        for L in seeds[common[0]]["layers"]:
            per_seed = []
            for s in common:
                a, c = seeds[s], ctrl[s]
                if step not in a["snaps"] or step not in c["snaps"]:
                    continue
                ca, cc = a["snaps"][step]["mlp"][L], c["snaps"][step]["mlp"][L]
                fa, fc = a["snaps"][step]["firing"][L], c["snaps"][step]["firing"][L]
                c0 = c["cos_pre"]["mlp"][L]
                if kind == "component":
                    df = fa - fc; top = c0 < np.percentile(c0, 20); bot = c0 > np.percentile(c0, 80)
                    per_seed.append(dict(rho_df_c0=spearmanr(df, c0)[0], df_top=df[top].mean(), df_bot=df[bot].mean(),
                                         cos_arm=ca.mean(), cos_ctrl=cc.mean(), loss_arm=a["snaps"][step]["loss"], loss_ctrl=c["snaps"][step]["loss"]))
                else:
                    g = a["groups"][L]; dc = ca - cc; df = fa - fc
                    per_seed.append(dict(dc_up=dc[g == 1].mean(), dc_none=dc[g == 0].mean(), dc_down=dc[g == -1].mean(),
                                         df_up=df[g == 1].mean(), df_down=df[g == -1].mean(),
                                         rho_df_dc=spearmanr(df[g != 0], dc[g != 0])[0], loss_arm=a["snaps"][step]["loss"], loss_ctrl=c["snaps"][step]["loss"]))
            if not per_seed:
                continue
            keys = per_seed[0].keys()
            row = {"layer": L, "n_seeds": len(per_seed)}
            for k in keys:
                v = np.array([d[k] for d in per_seed], float)
                row[k] = float(v.mean()); row[k + "_sem"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            rows.append(row)
        if rows:
            cols = [c for c in rows[0] if not c.endswith("_sem")]
            lines.append(f"\n### snapshot step {step} (intervention at {seeds[common[0]]['intervene_at']})\n")
            lines.append(md_table(rows, cols))
            sem_cols = ["layer"] + [c for c in rows[0] if c.endswith("_sem")]
            lines.append("\ns.e.m. over seeds:\n" + md_table(rows, sem_cols, fmt="{:.4f}"))
            tables[f"{arm}@{step}"] = rows

# --- pooled dose-response over all bias arms, last snapshot
bias_arms = [a for a in runs if a.startswith("bias:")]
if bias_arms:
    lines.append("\n## Dose-response pooled over bias arms (last snapshot, all intervened units, all seeds)\n")
    for L in runs[bias_arms[0]][next(iter(runs[bias_arms[0]]))]["layers"]:
        dfs, dcs, c0s = [], [], []
        for arm in bias_arms:
            for s, a in runs[arm].items():
                if s not in ctrl:
                    continue
                c = ctrl[s]; step = snaps_after(a)[-1]
                if step not in c["snaps"]:
                    continue
                g = a["groups"][L]; m = g != 0
                first = snaps_after(a)[0]
                dfs.append((a["snaps"][first]["firing"][L] - c["snaps"][first]["firing"][L])[m])   # realized, immediate window
                dcs.append((a["snaps"][step]["mlp"][L] - c["snaps"][step]["mlp"][L])[m]); c0s.append(c["cos_pre"]["mlp"][L][m])
        if dfs:
            d = dose_response(np.concatenate(dfs), np.concatenate(dcs), np.concatenate(c0s))
            lines.append(f"- layer {L}: linear slope {d['linear_slope']:+.3f} (R² {d['linear_r2']:.3f}); nonlinear "
                         f"[Δf, Δf², c₀Δf] = {np.round(d['nonlin'], 3).tolist()} (R² {d['nonlin_r2']:.3f}); Spearman {d['spearman']:+.2f}\n")
            tables[f"dose@L{L}"] = d

# --- growth law on controls
lines.append("\n## Growth law on control runs: Δc = a (f − f̄) + b c + const (intervene_at → last snapshot)\n")
for L in ctrl[next(iter(ctrl))]["layers"]:
    vals = []
    for s, c in ctrl.items():
        first, last = snaps_after(c)[0], snaps_after(c)[-1]
        vals.append(growth_law(c["snaps"][first]["firing"][L], c["cos_pre"]["mlp"][L], c["snaps"][last]["mlp"][L] - c["cos_pre"]["mlp"][L]))
    a = np.array([v["a"] for v in vals]); b = np.array([v["b"] for v in vals]); pr = np.array([v["partial_f_dc_given_c"] for v in vals])
    lines.append(f"- layer {L}: a = {a.mean():+.3f} ± {a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0:.3f}, b = {b.mean():+.3f}, partial ρ(f, Δc | c) = {pr.mean():+.2f}  (n={len(a)})\n")

od = outdir("05_analysis")
open(os.path.join(od, "report.md"), "w").write("".join(lines))
save_json(tables, os.path.join(od, "tables.json"))
print("".join(lines))
