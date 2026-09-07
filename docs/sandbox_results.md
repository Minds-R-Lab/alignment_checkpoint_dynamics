# Overnight results — read/write alignment: overlap check + Stage 1 checkpoint dynamics

Date: 6 Sep 2026. Model: EleutherAI/pythia-70m, 16 public checkpoints (steps 0, 1, 16, 64, 256, 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 100k, 143k). Batch = 2M tokens/step; 143k steps = 300B tokens.

## 0. Overlap check on the one flagged paper

arXiv:2605.16600 ("Where Pretraining writes and Alignment reads", May 2026) tracks how *weight deltas* align with residual-stream activation subspaces and the unembedding's prediction subspace (a "relative-subspace-fraction" probe). It does **not** measure per-neuron cos(write, read). Gap 1 remains open. Its mechanism — "updates to W are sums of outer products δₜaₜᵀ and inherit directional structure from whichever side has concentrated covariance" — is the same mechanism as our M = E[g xᵀ] argument, so it is a supporting citation, not a competitor.

## 1. Per-layer MLP mean cos(write, read) across training

| step | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| 0–16 | 0.000 | 0.000 | 0.000 | −0.001 | 0.000 | −0.001 |
| 64 | −0.001 | −0.002 | −0.003 | −0.003 | −0.001 | −0.002 |
| 256 | −0.006 | −0.033 | −0.030 | −0.022 | −0.013 | −0.006 |
| 512 | −0.009 | −0.054 | −0.051 | −0.041 | −0.028 | −0.015 |
| 1000 | −0.013 | −0.103 | −0.121 | −0.133 | −0.061 | −0.027 |
| 2000 | +0.003 | −0.183 | −0.244 | −0.229 | −0.166 | −0.059 |
| 4000 | +0.023 | −0.245 | −0.308 | −0.257 | −0.203 | −0.091 |
| 8000 | +0.041 | −0.250 | −0.327 | −0.261 | −0.213 | −0.120 |
| 16000 | +0.056 | −0.253 | −0.342 | −0.262 | −0.217 | −0.147 |
| 32000 | +0.068 | −0.263 | −0.360 | −0.268 | −0.217 | −0.159 |
| 64000 | +0.077 | −0.274 | −0.379 | −0.271 | −0.207 | −0.172 |
| 100000 | +0.080 | −0.274 | −0.374 | −0.255 | −0.158 | −0.158 |
| 143000 | +0.071 | −0.269 | −0.364 | −0.245 | −0.144 | −0.149 |

Permuted-pair control: |value| ≤ 0.005 at every checkpoint and layer.

Readout:
- Zero through step 16. First detectable at step 64 (~130M tokens). Steep rise 256 → 4000. Layers 1–3 reach 90% of final by step 8000 (5.6% of training). Layer 5 is slower (75% at 8k, saturates ~64k).
- Layer 4 overshoots (−0.217 at 16k–32k) and relaxes to −0.144 by the end.
- **Layer 0 flips sign during training**: weakly negative through step 1000 (−0.013), then positive from step 2000, growing to +0.08. Layer 0 in Pythia reads raw embeddings (parallel attention/MLP), and its sign is decided after the other layers'.

## 2. The mean and the concentration have different timescales (new result)

Fraction of units with cos < −0.5 (the strongly self-adjoint minority):

| step | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| ≤2000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| 4000 | 0.027 | 0.055 | 0.019 | 0.020 | 0.006 |
| 8000 | 0.074 | 0.128 | 0.073 | 0.075 | 0.021 |
| 16000 | 0.084 | 0.194 | 0.119 | 0.129 | 0.042 |
| 32000 | 0.092 | 0.268 | 0.190 | 0.181 | 0.058 |
| 64000 | 0.109 | 0.349 | 0.246 | 0.213 | 0.072 |
| 100000 | 0.130 | 0.375 | 0.243 | 0.163 | 0.061 |
| 143000 | 0.134 | 0.363 | 0.233 | 0.148 | 0.045 |

Distribution of per-unit cos in layers 2 and 3 (p10 / p50 / p90, std), with the hidden-basis-rotation std for comparison:

| step | layer | mean | std | p10 | p50 | p90 | frac<−0.5 | rotated-basis std | tr(block)/‖block‖_F |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | L2 | −0.244 | 0.099 | −0.366 | −0.251 | −0.115 | 0.000 | 0.042 | −8.00 |
| 8000 | L2 | −0.327 | 0.153 | −0.515 | −0.338 | −0.123 | 0.128 | 0.042 | −10.50 |
| 32000 | L2 | −0.360 | 0.195 | −0.598 | −0.382 | −0.091 | 0.268 | 0.048 | −10.98 |
| 143000 | L2 | −0.364 | 0.244 | −0.649 | −0.410 | −0.007 | 0.363 | 0.051 | −10.30 |
| 2000 | L3 | −0.229 | 0.097 | −0.348 | −0.232 | −0.104 | 0.000 | 0.042 | −7.75 |
| 8000 | L3 | −0.261 | 0.176 | −0.473 | −0.278 | −0.019 | 0.073 | 0.040 | −9.40 |
| 32000 | L3 | −0.268 | 0.245 | −0.561 | −0.294 | +0.076 | 0.190 | 0.042 | −9.30 |
| 143000 | L3 | −0.245 | 0.294 | −0.605 | −0.264 | +0.170 | 0.233 | 0.042 | −7.55 |

Readout — two phases:
- **Phase 1 (≈ first 5%): a uniform shift.** At step 2000 every unit is mildly anti-aligned (p10 −0.37, p90 −0.12, std 0.10, nobody below −0.5). This is close to what a rotated basis would show (std 0.04): the effect is essentially the spectral part, tr(W_down W_up), spread evenly. This is exactly what the linearized two-factor dynamics predict (every unit's write vector grows along −M·a).
- **Phase 2 (the remaining 95%): polarization at fixed mean.** From 8k to 143k the mean barely moves (L2 −0.33 → −0.36; L3 −0.26 → −0.25) while the std doubles (0.15 → 0.24/0.29), p10 drops to −0.65, and p90 rises to 0 or +0.17. A minority of units becomes strongly self-adjoint; others drift to zero or even positive. The composed block map's trace (spectral) is flat or declining in this phase; the non-spectral, basis-dependent concentration is what grows.

So the two components that the rotation control separated in weight space are also separated in time: the spectral component is an early optimization-induced effect; the concentration onto specific units develops on the capability timescale.

## 3. Attention OV copy-alignment on the same checkpoints

Mean cos(value-input row, output-projection column) per channel:

| step | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| ≤64 | +0.003 | +0.003 | +0.002 | +0.002 | −0.001 | −0.003 |
| 1000 | +0.004 | −0.029 | −0.036 | −0.064 | −0.125 | +0.014 |
| 2000 | +0.011 | −0.018 | −0.041 | −0.067 | −0.174 | +0.141 |
| 4000 | +0.012 | −0.030 | −0.069 | −0.030 | −0.145 | +0.267 |
| 8000 | +0.012 | −0.040 | −0.079 | +0.002 | −0.103 | +0.323 |
| 16000 | +0.009 | −0.048 | −0.079 | +0.021 | −0.051 | +0.342 |
| 32000 | +0.009 | −0.048 | −0.077 | +0.040 | −0.015 | +0.377 |
| 64000 | +0.010 | −0.035 | −0.068 | +0.056 | +0.005 | +0.445 |
| 100000 | +0.011 | −0.006 | −0.044 | +0.067 | +0.114 | +0.535 |
| 143000 | +0.008 | −0.010 | −0.043 | +0.070 | +0.127 | +0.538 |

Readout:
- The final layer's copy-alignment **keeps growing through the entire run** (0.27 at 4k → 0.34 at 16k → 0.45 at 64k → 0.54 at 143k), with a large late jump between 64k and 100k. This does *not* plateau the way induction-head scores do; the literature's plateau consensus is about a different observable (in-context copying behaviour), not this weight-space quantity. This is a reportable finding in its own right.
- Layer 4 OV flips sign: deletion-type (−0.174 at 2k) → copy-type (+0.127 at the end). Layers 1–2 stay weakly deletion-type.

## 4. What this adds to the paper plan (relative to the literature report)

Gap 1 (training dynamics): now measured at 16 checkpoints with early log-spaced resolution. Gap 3 (controls): permutation control at every checkpoint, rotation control at four. The new, unanticipated finding is the two-timescale split in §2, which ties Gaps 1 and 3 together: the spectral/trace part emerges early and uniformly (consistent with the two-factor linearized dynamics), the non-spectral concentration develops slowly at fixed mean. The OV result in §3 corrects the report's caution: weight-space copy-alignment grows monotonically here.

## 5. Next steps (in priority order)

1. Replicate §1–§3 on pythia-160m and pythia-410m checkpoints (same script, `ckpt_metrics.py`, adjust layer count/head dims) to check that both timescales and the layer-0 sign flip are family-invariant.
2. OLMo checkpoints (gated SwiGLU) to test whether the early *sign* differs from Pythia's from the very first checkpoints (predicted by the q = aᵀMa mechanism) and whether Phase 2 polarization is architecture-independent.
3. Per-unit tracking: follow individual units from step 4k to 143k — do the strongly self-adjoint units at the end come from the most-aligned units at 8k (deepening), or is there churn (units enter/exit the tail)? This distinguishes "rich get richer" from re-organization.
4. Tie the Phase-2 polarization to activation frequency (Gurnee et al. found high-frequency units are the anti-aligned ones): if the tail units are the high-frequency units, Phase 2 is the formation of frequently-firing "cleanup" units.
5. Theory: Phase 1 is the linearized regime; Phase 2 needs a nonlinear argument (e.g., units specializing via winner-take-all under weight decay). Weight decay is the likely driver of the late relaxation in layers 3–4 and of the tr/‖·‖ decline.

Scripts and data in /home/claude/spec: `ckpt_metrics.py` (per-checkpoint metrics), `ckpt_results.json` (all numbers above), `train_sign.py` (toy models), `nonspec.py` (spectral-surrogate battery).

---

# Addendum (7 Sep): mechanism experiments on the Phase-2 polarization

## A. Unit trajectories: rich-get-richer, no reassignment
Per-unit cos at step 8k vs final (143k), Pythia-70m:

| layer | corr(c8k, c143k) | corr(c32k, c143k) | final tail (c<−0.5) already in most-aligned 20% at 8k / 32k | 8k-leaders that end in tail | 8k-leaders that end near 0 |
|---|---|---|---|---|---|
| 1 | 0.87 | 0.95 | 80% / 93% | 54% | 0% |
| 2 | 0.84 | 0.94 | 51% / 54% | 93% | 0% |
| 3 | 0.89 | 0.95 | 68% / 76% | 79% | 0% |
| 4 | 0.90 | 0.96 | 85% / 93% | 63% | 0% |
| 5 | 0.83 | 0.93 | 85% / 93% | 19% | 1% |

The ordering of units is fixed by ~8k (Phase 1); Phase 2 deepens along it. No churn.

## B. Activation frequency: geometry precedes function
f = P(pre-activation > 0) over 24k tokens (Shakespeare; out-of-distribution caveat). Spearman:

| layer | rho(f, c) at 8k | rho(f, c) at 143k | partial rho(c8k → f143k \| f8k) | partial rho(f8k → c143k \| c8k) | median f: tail vs others (143k) | Δf 8k→143k, tail |
|---|---|---|---|---|---|---|
| 1 | −0.41 | −0.56 | −0.29 | −0.07 | 0.22 vs 0.11 | −0.01 |
| 2 | −0.40 | −0.66 | −0.47 | −0.04 | 0.28 vs 0.12 | +0.03 |
| 3 | −0.56 | −0.77 | −0.51 | −0.04 | 0.55 vs 0.15 | +0.18 |
| 4 | −0.38 | −0.69 | −0.59 | −0.02 | 0.62 vs 0.14 | +0.26 |
| 5 | +0.10 | +0.11 | — | — | 0.02 vs 0.18 | −0.02 |

Final cos is monotone in frequency (layer 2, deciles rarest→most frequent: −0.06, −0.19, −0.24, −0.32, −0.37, −0.40, −0.44, −0.47, −0.54, −0.61). Cross-lagged partials: early geometry predicts later firing rate; early firing rate does not predict later geometry. Layer 5 (final layer) is a different regime: its few tail units are rare-firing.

## C. Load-bearing ablation (final checkpoint, layers 1–5, same 24k tokens)
| intervention on write vectors d_k | loss |
|---|---|
| baseline | 4.039 |
| remove aligned component d_k − (d_k·â_k)â_k (10–20% of energy) | **11.896** |
| remove a random orthogonal component of the same norm per unit | 4.570 |
| flip sign of aligned component | 25.861 |
| keep only the aligned component | 18.494 |

The self-adjoint component is catastrophically load-bearing relative to its energy (worse than uniform random prediction when removed), consistent with a negative-feedback role that keeps the residual stream bounded.

## D. Closed leads
- Layer-0 T ≈ f(C_E): closed negative (commutator equals null for all embedding-derived candidates).
- Hidden permutation / cross-unit correspondence: closed negative (Hungarian at chance).

## E. Updated research question
Not "why do read/write vectors anti-align" but: why does training first install a near-uniform self-adjoint (negative-feedback) component across all units in the linearized regime, then break that symmetry by deepening it in the units that go on to fire most — and why is that small component the one the network cannot live without?

## F. Radial-contraction test (Pythia-70m final, 24k tokens)
E[(mlp_out · h)/‖h‖²] per layer (negative = contraction along the residual direction), and residual norm entering each layer:

| model | loss | radial L0..L5 | residual norm at input of L0..L5 |
|---|---|---|---|
| original | 4.08 | +0.33 −0.10 −0.18 −0.38 −0.40 +0.08 | 0.6 6.7 8.2 11.5 13.6 15.9 |
| aligned component removed (L1–5) | 12.07 | +0.33 +0.07 +0.13 +0.02 +0.05 +0.01 | 0.6 6.7 9.2 14.3 20.1 37.6 |
| aligned component flipped | 25.93 | +0.33 +0.24 +0.37 +0.31 +0.51 +0.23 | 0.6 6.7 10.6 18.5 32.1 85.1 |

The self-adjoint component is the radial contraction; the ablation catastrophe is a norm blow-up. "Negative feedback" is now established, not assumed.

## G. Common-cause test: FALSIFIED
Read-direction variance vᵢ = âᵢᵀCâᵢ of the MLP input (143k) does not predict alignment (ρ = +0.14, +0.25, −0.07, −0.21 in L1–4) or firing rate (ρ = −0.03, −0.16, +0.07, +0.26), and the early-geometry → later-frequency partial survives controlling for it (−0.49 to −0.70).

## H. Interventional test (toy Pythia-style model, one seed, identical data order)
At step 700, aligned component of write vectors in layers 1–2 tripled (amp), removed (rem), or unchanged (ctrl); firing measured over steps 1120–1150.

| layer | mean cos ctrl / amp / rem | mean f ctrl / amp / rem | Spearman(Δf_amp, c) | Spearman(Δf_rem, c) | Δf most-aligned 20% (amp, rem) | Δf least-aligned 20% (amp, rem) |
|---|---|---|---|---|---|---|
| 1 | −0.150 / −0.204 / −0.116 | 0.204 / 0.205 / 0.202 | −0.29 | +0.51 | (+0.008, −0.008) | (−0.003, +0.003) |
| 2 | −0.130 / −0.189 / −0.091 | 0.231 / 0.228 / 0.230 | −0.17 | +0.47 | (+0.000, −0.008) | (−0.006, +0.004) |

Losses identical (1.540) in all three runs. Findings: (i) the geometry → firing arm is causal (removing the component lowers later firing of the most-aligned units; tripling raises it; layer mean unchanged — redistribution); (ii) the component is an attractor: after removal it regrew to 75% of control within 450 steps. Effect sizes are small and need multi-seed replication on the real model (intervene at step 8k, continue pretraining).

## I. Open
1. Mechanism of the polarization ordering (what sets the unit ranking by ~8k). Tested and failed: read-direction variance.
2. Return arm f → c of the proposed feedback loop: intervene on firing rate (bias shift on a unit subset), measure later geometry.
3. Replication of A–H on pythia-160m/410m checkpoints and on OLMo (gated; sign question).

## J. Reverse intervention f → c (toy, one seed, matched data order)
At step 700, a random 20% of units per layer (layers 1–2) had their up-projection bias shifted by +0.73 (≈ +1 std of pre-activation) and another random 20% by −0.73; biases stayed trainable. Compared against an identical control run.

| layer | group | firing pre (670–700) ctrl/bias | firing post (701–730) ctrl/bias | firing 1120–1150 ctrl/bias | c at 1150 ctrl / bias | Δc (bias − ctrl) |
|---|---|---|---|---|---|---|
| 1 | up | 0.265/0.265 | 0.258/0.642 | 0.202/0.295 | −0.131 / −0.144 | −0.013 ± 0.005 |
| 1 | none | 0.266/0.266 | 0.260/0.259 | 0.204/0.197 | −0.155 / −0.147 | +0.008 ± 0.001 |
| 1 | down | 0.262/0.262 | 0.256/0.053 | 0.204/0.140 | −0.154 / −0.098 | +0.056 ± 0.005 |
| 2 | up | 0.284/0.284 | 0.276/0.691 | 0.232/0.362 | −0.141 / −0.165 | −0.024 ± 0.005 |
| 2 | none | 0.282/0.282 | 0.276/0.278 | 0.230/0.224 | −0.130 / −0.124 | +0.006 ± 0.001 |
| 2 | down | 0.282/0.282 | 0.275/0.066 | 0.233/0.151 | −0.118 / −0.063 | +0.055 ± 0.005 |

Per-unit Spearman(Δf_post, Δc) = −0.47 (L1), −0.51 (L2). Firing rate causally sets the strength of the self-adjoint component; combined with section H, c ↔ f is a closed feedback loop.

Growth-rate law fitted on the control run, Δcᵢ(700→1150) = a(fᵢ − f̄) + b·cᵢ: a = −0.22 (L1), −0.09 (L2); b ≈ 0 (R² small on the natural variation; the intervention shows the same sign with large effect). The instability is mediated by firing, not by c amplifying itself. Because the loop redistributes firing (Σ(fᵢ − f̄) = 0), the mean of c is conserved to first order while its variance grows — the Phase-2 signature, derived. Asymmetry: suppression changes c ~3× more than enhancement; a linear law is only a first approximation.

Preregistered real-model version: Pythia-70m at step 8k, random-unit bias shifts of ±0.5/±1/±2 std, ≥5 seeds, continued pretraining on the Pile, controls as in GPT's table; measure Δc at +2k, +8k, +24k steps.

## K. Dose-response of the reverse intervention (toy, one seed; same random unit groups)

| dose (std) | layer | group | realized Δf (701–730) | Δf (1120–1150) | Δc ± se |
|---|---|---|---|---|---|
| 0.5 | 1 | up | +0.197 | +0.043 | −0.012 ± 0.003 |
| 0.5 | 1 | down | −0.137 | −0.036 | +0.026 ± 0.003 |
| 1.0 | 1 | up | +0.385 | +0.093 | −0.013 ± 0.005 |
| 1.0 | 1 | down | −0.203 | −0.064 | +0.056 ± 0.005 |
| 2.0 | 1 | up | +0.625 | +0.244 | +0.006 ± 0.007 |
| 2.0 | 1 | down | −0.247 | −0.122 | +0.115 ± 0.008 |
| 0.5 | 2 | up | +0.211 | +0.059 | −0.020 ± 0.003 |
| 0.5 | 2 | down | −0.135 | −0.047 | +0.028 ± 0.003 |
| 1.0 | 2 | up | +0.416 | +0.130 | −0.024 ± 0.005 |
| 1.0 | 2 | down | −0.210 | −0.082 | +0.055 ± 0.005 |
| 2.0 | 2 | up | +0.645 | +0.336 | −0.002 ± 0.007 |
| 2.0 | 2 | down | −0.264 | −0.143 | +0.100 ± 0.009 |

Pooled fits (n = 1224 intervened units): linear in realized immediate Δf, R² 0.22 (slope −0.10); linear in imposed signed dose, R² 0.29; Δc = −0.42·Δf_late + 0.89·Δf_late² − 0.27·c₀·Δf_late, R² 0.36. Spearman(realized Δf, Δc) = −0.51.

Findings: (i) suppression response is monotone in dose and large; (ii) enhancement response saturates and vanishes at the largest dose — the response function is concave in firing; (iii) realized P(z>0) is not the sufficient statistic (imposed dose predicts Δc better because realized Δf floors at zero firing) — the mechanistic variable is likely the unit's gradient exposure E[GELU(z)·(g·â)], which the real-model experiment should measure directly; (iv) concavity implies a bounded polarization (instability strongest for rare-firing units, dying for frequent ones), a candidate explanation for the empirical |cos| ≲ 0.8 ceiling (Gurnee et al. 2024), and only approximate conservation of the mean.
