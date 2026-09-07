# Preregistration — real-model tests of the read/write feedback loop

Written before any Stage-4 run on real models. Numbers in `docs/sandbox_results.md` (Pythia-70m
checkpoints; toy interventions, one seed) are the priors; they are NOT evidence for the hypotheses below.

## Definitions
* `c_k(t)` = cos(read_k, write_k) at step t; `f_k(t)` = P(pre-activation_k > 0) over the `window` steps
  ending at t; `e_k(t)` = gradient exposure E[h_k (g·â_k)] over the same window.
* An "arm" is a continued-pretraining run from a fixed checkpoint (default Pythia-70m `step8000`) with one
  intervention applied at step `intervene_at` after a short warm-up; its "control" is the run with the
  same `--seed` and no intervention. All comparisons are arm − control on the same seed, then averaged
  over seeds with s.e.m. (n = 5 seeds minimum).

## H1 — trajectory stability (Stage 2, observational)
Prediction: Spearman(c(step8k), c(final)) ≥ 0.7 in every layer ≥ 1, and no unit in the most-aligned 20 %
at 8k ends with |c| < 0.1. Falsified if corr < 0.5 in a majority of layers or churn > 10 %.

## H2 — geometry precedes firing (Stage 2, observational)
Prediction: partial ρ(c_early → f_late | f_early) < −0.2 in every layer 1..L−2, and
|partial ρ(f_early → c_late | c_early)| < 0.1. Falsified if the reverse ordering holds in a majority of layers.

## H3 — load-bearing radial contraction (Stage 3)
Prediction: loss(remove_aligned) − loss(original) ≥ 10 × [loss(remove_random) − loss(original)] averaged
over 3 random seeds; radial projection sign flips from negative (original) to ≥ 0 (removed) in layers
with mean c < −0.1; loss(flip) > loss(remove). Falsified if the aligned/random loss ratio < 3.

## H4 — c → f (Stage 4, arm `component:<α>`)
Prediction at the +500 and +2000 snapshots: Spearman(Δf, c_pre) > 0.3 for α = −1 (removal lowers the
later firing of the most-aligned units) and < −0.1 for α = +2; Δf of the most-aligned 20 % has the
predicted sign in ≥ 4/5 seeds; layer-mean f unchanged (|Δf̄| < 0.01). Regrowth: for α = −1, c at +2000
recovers ≥ 50 % of the control value (attractor). Falsified if the Spearman has the wrong sign in ≥ 3/5
seeds or |Spearman| < 0.1 at both snapshots.

## H5 — f → c (Stage 4, arm `bias:<β>:random`, β ∈ {0.5, 1, 2})
Randomly selected 20 % "up" and 20 % "down" groups; biases trainable.
Prediction at +500 and +2000: Δc(down) > 0 and Δc(up) ≤ 0 with Δc(down) − Δc(up) > 0.02 at β = 1 in
≥ 4/5 seeds; per-unit Spearman(realized Δf, Δc) < −0.2; monotone dose-response on the down side
(Δc(β=2) > Δc(β=1) > Δc(β=0.5)); saturation on the up side is allowed and expected. Also record whether
the imposed β or the realized Δf (or Δe) is the better predictor of Δc — the sandbox found the imposed
bias predicted better, implying `P(z>0)` is not the sufficient statistic; we predict `e` will be.
Falsified if Δc(down) − Δc(up) ≤ 0 in ≥ 3/5 seeds at both β = 1 and β = 2.

## H6 — conservation and variance growth (Stage 1/2, observational)
Prediction across Pythia sizes: from the checkpoint where mean c reaches 80 % of final, the mean changes
by < 25 % while the fraction of units with c < −0.5 at least doubles. Falsified if the mean keeps
growing proportionally with the tail.

## Controls that must be reported alongside every result
* permutation control (destroys correspondence) — expected ≈ 0
* hidden-basis rotation control — expected: Σ a_k·d_k exactly unchanged, mean c approximately unchanged,
  tail fraction → ≈ 0
* spectral surrogates (same singular values, random singular vectors) for any "z-score vs spectrum" claim
* norm-matched random removal for every ablation
* `bias:<β>:aligned` (groups selected by alignment) may be run for comparison but is confounded by
  regression to the mean and is NOT the preregistered test

## What would make us abandon the loop hypothesis
H4 and H5 both falsified on Pythia-70m with 5 seeds. What would make us call it architecture-specific:
H4/H5 hold on Pythia but H2 or H5 fail on a gated model (OLMo-2-1B or SmolLM2) run with the same protocol.
