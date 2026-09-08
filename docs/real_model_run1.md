# First real-model run (quick mode, one seed) — verdicts against PREREGISTRATION.md

Pythia-70m, continued pretraining from `step8000` on the Pile (32×2048 tokens/step, lr 3e-4, bf16),
intervention at step 200, snapshots at +30/+500/+1800/+2000/+8000. In-distribution eval text throughout.
OLMo-2-1B checkpoints for the gated-family sweep. Date: 8 Sep 2026.

## H1 trajectory stability — HOLDS
corr(c_8k, c_final) = 0.83–0.90 in layers 1–5; churn of early leaders = 0 % (layer 5: 1 %).

## H2 geometry precedes firing — NOT SUPPORTED on in-distribution text
On Pile text, Spearman(firing, cos) is already −0.85 to −0.93 at step 8k in layers 1–4 (it was −0.4 to
−0.6 on Shakespeare in the sandbox). With the two rankings nearly identical at the early checkpoint, the
cross-lagged partials lose their leverage and come out mixed: geom→fire = −0.01, −0.14, −0.28, −0.40 and
fire→geom = −0.33, −0.22, −0.25, −0.08 (L1–L4). The sandbox asymmetry was an artifact of measuring firing
out of distribution. To test ordering, the early checkpoint must be taken while the alignment is still
forming (steps 512–4000), not at 8k. **Action:** rerun 02 with `--revisions step1000 step2000 step4000 step143000`.

## H3 load-bearing radial contraction — HOLDS, strongly
| intervention (layers 1–5) | loss | radial L1..L5 | residual norm into L5 |
|---|---|---|---|
| original | 2.758 | −0.10 −0.20 −0.43 −0.41 −0.19 | 15.2 |
| remove aligned | 14.10 | +0.07 +0.11 −0.01 +0.18 −0.14 | 45.6 |
| remove random, norm-matched (3 seeds) | 3.42–3.49 | unchanged | 16.0–16.3 |
| flip aligned | 30.10 | +0.25 +0.34 +0.30 +0.72 +0.19 | 103.4 |
| keep only aligned | 19.01 | | 16.5 |
Aligned/random loss-increase ratio = 16 (criterion ≥ 10). Radial sign flips in layers 1–4; layer 5 stays
slightly negative. Single-layer removal: layer 4 costs most (4.75), then layer 3 (3.35).

## H4 c → f — MIXED; the attractor result is the strong part
* Removal (α = −1): immediately after removal the most-aligned 20 % of units fire 0.04–0.05 less
  (Spearman(Δf, c_pre) = +0.50 to +0.59), then the effect fades as the component regrows. Regrowth:
  layer-2 mean cos −0.02 → −0.12 (+500) → −0.21 (+1800) → −0.30 (+8000) vs control −0.35: **88 % recovered**.
  The component is a dynamical attractor in the real model.
* Tripling (α = +2): the OPPOSITE of the toy prediction. At +500, Spearman(Δf, c_pre) = +0.93 in every
  layer: the most-aligned units fire 0.10–0.14 LESS. Reading: all layers were intervened at once, so a
  tripled suppression write in layer L lowers the same feature for units in layer L+1 that read it — a
  within-forward-pass coupling across layers, not a training-dynamics effect. The tripled component also
  relaxes back toward the control (−0.65 → −0.43 by +8000). H4's α = +2 clause is falsified as stated;
  the causal arm c → f, if it exists in the real model, is not "aligned units fire more later".
  **Action:** intervene on ONE layer at a time and measure firing in that layer vs downstream layers.

## H5 f → c — HOLDS at β = 2; vacuous at β ≤ 1 because trainable biases were optimized away
| β | realized Δf(down) at +30 / +500 / +8000 | Δc(down) − Δc(up) at +2000 (L1..L5) | at +8000 |
|---|---|---|---|
| 0.5 | −0.16 / −0.02 / −0.01 | +.006 +.008 +.002 +.003 −.000 | +.011 +.016 +.005 +.009 +.003 |
| 1 | −0.18 / −0.04 / −0.02 | +.006 +.012 +.003 +.004 −.002 | +.016 +.027 +.007 +.015 +.004 |
| 2 | −0.20 / −0.16 / −0.14 | +.036 +.052 +.043 +.028 +.005 | +.105 +.104 +.083 +.075 +.036 |
At β = 2 the shift persisted and the geometric response has the predicted sign, grows over the run, and
correlates per unit with the realized firing change at ρ = −0.65 to −0.71 (L1–L2), −0.48/−0.58 (L3–L4).
At β ≤ 1 the optimizer removed the bias within ~500 steps, so the dose was ≈ 0 — the preregistered β = 1
threshold fails for a reason that is procedural, not physical. Amendment: `bias:<β>:random:frozen` arms.
Predictor comparison at β = 2: realized Δf at +30 (ρ −0.67/−0.71) ≈ imposed dose (−0.64/−0.69);
gradient exposure Δe at +30 does NOT predict Δc (+0.00/+0.49). The preregistration's guess that exposure
is the sufficient statistic is falsified; firing rate is at least as good as the imposed dose here.

## H6 conservation and variance growth — HOLDS (Pythia-70m checkpoints)
Layer 2: mean −0.31 at 4k → −0.36 at 143k (+16 %) while frac(c < −0.5) goes 0.06 → 0.36 (6×).

## Gated family (OLMo-2-1B, 16 layers) — the sign puzzle resolves
| checkpoint | mean cos(write, up) by layer 0..15 |
|---|---|
| step 10k (21B tokens) | −.08 +.08 +.02 −.05 −.15 −.19 −.21 −.25 −.29 −.30 −.27 −.25 −.20 −.17 −.15 −.14 |
| step 50k (105B) | +.20 +.28 +.22 +.21 +.08 +.02 −.04 −.09 −.18 −.21 −.17 −.17 −.12 −.13 −.16 −.20 |
| step 150k (315B) | +.43 +.37 +.35 +.36 +.23 +.22 +.14 +.11 −.01 −.06 −.03 −.04 −.01 −.07 −.16 −.23 |
| final | +.37 +.41 +.44 +.33 +.32 +.20 +.12 −.01 −.08 −.01 −.08 −.11 −.07 −.15 −.23 −.27 |
Early in training the gated model is NEGATIVE almost everywhere, like Pythia — consistent with the
Phase-1 mechanism having one sign. The positive ("enrichment") alignment of early layers is a LATE,
capability-timescale development that overwrites it. The Gerstner–Schütze sign difference between GeLU
and gated models is a difference in what happens after Phase 1, not in Phase 1.

## Code issues found by this run (fixed in this version)
1. `04_intervene.py` computed `eval_loss` after removing the virtual biases — the bias arms' eval losses
   (3.2 / 4.0 / 4.65 vs 3.03) are meaningless; training losses (3.10 vs 3.09 at +8000) are the valid comparison.
   Now evaluates both with and without the biases.
2. `--steps` was ignored (total = max snapshot); fixed.
3. Trainable biases are optimized away at small doses; added `:frozen` arms and amended the preregistration.

## What changed in the picture
Confirmed on the real model: unit ordering fixed early; the aligned component is a dynamical attractor,
is the radial contraction, and is load-bearing 16× beyond a norm-matched control; suppressing a unit's
firing weakens its self-adjoint component with a dose-dependent, per-unit-correlated response; the mean is
conserved while the tail grows; and the gated-model sign flip is a late overwrite of a common early sign.
Retracted or reframed: "geometry precedes firing" (untestable at 8k on real text; must be measured while
alignment forms); the toy's c→f sign (the real model shows an immediate cross-layer coupling of the
opposite sign). The loop is therefore half-confirmed (f→c) and half-open (c→f), with the attractor
property the most robust dynamical fact.
