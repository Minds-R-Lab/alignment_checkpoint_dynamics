# rwloop — read/write self-alignment of transformer hidden units

Code to test, on real models and with matched-seed interventions, the following claims (developed on
Pythia-70m and a toy transformer; see `docs/sandbox_results.md` for the numbers this repo is designed
to replicate or refute):

1. **Phase 1.** In the first few percent of pretraining, every MLP hidden unit acquires a component of its
   write vector along its own read vector, `d_k ≈ λ_k a_k + r_k`, with a layer-dependent sign
   (negative = suppression/"depletion", positive = enrichment). The sign is predicted by the
   gradient–input quadratic form `q = a^T E[g x^T] a` (two-factor linearized dynamics).
2. **Phase 2.** For the rest of training the *mean* alignment is roughly conserved while the per-unit
   distribution polarizes onto a minority of strongly self-adjoint units, along an ordering that is
   fixed early (no reassignment).
3. **Two kinds of quantity.** The mean is (approximately) a spectral quantity — the exact invariant is
   `Σ_k a_k·d_k = tr(W_down W_up)`, unchanged by any rotation of the hidden basis — while the
   concentration is not: it exists only in the coordinate system the nonlinearity fixes. Permuting the
   unit correspondence destroys both; rotating the hidden basis destroys only the concentration.
4. **Feedback loop.** Manipulating the aligned component changes which units later fire most
   (`c → f`), and manipulating firing changes the later aligned component (`f → c`), with a concave,
   saturating response and a dose-dependent effect.
5. **Function.** The aligned component is the MLP's radial contraction of the residual stream and is
   catastrophically load-bearing relative to its 10–20 % share of write energy.

The static observation (1, without the sign mechanism) is published: Gerstner & Schütze 2025
(arXiv:2505.17936), Gurnee et al. 2024 (arXiv:2401.12181), Elhage et al. 2021. Items 2–5 and the
training-dynamics content of 1 are, as of the literature sweep in `docs/literature.md`, unclaimed.

## Install

```bash
git clone <this repo> && cd rwloop
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest
pytest -q            # 18 tests on tiny random models, ~30 s, no downloads
```

Python ≥ 3.10, CUDA GPU for scripts 02–04 (one H100 is plenty), CPU is fine for 01 and 06.
Data: the default corpus is `monology/pile-uncopyrighted` (Pythia's distribution), streamed via
`datasets`; the eval set (1 M tokens) is cached under `data_cache/`. Set `RWLOOP_CACHE` to move it.

## Layout

```
rwloop/adapters.py    uniform access to read/write/gate/V/O weights and hook targets (gpt_neox, llama-like)
rwloop/metrics.py     per-unit cos, aligned energy, block trace, tail fractions; permutation / rotation /
                      spectral-surrogate controls; attention OV cos
rwloop/hooks.py       firing rate, gradient exposure E[h_k (g·â_k)], radial projection, residual norms
rwloop/intervene.py   ablations (remove / flip / keep-only / norm-matched random), component scaling,
                      VirtualBias (trainable, works for gated models with no bias)
rwloop/train.py       continued pretraining with matched data order and per-unit snapshots
rwloop/analysis.py    cross-lagged partial correlations, trajectory stability, dose-response, growth law
scripts/01..06        the experiments (below)
tests/test_all.py     each test names the bug it catches; three deliberate bugs verified to fail it
docs/                 sandbox results (baseline expectations), literature sweep, preregistration
```

## Experiments

| # | script | question | cost |
|---|---|---|---|
| 1 | `01_checkpoint_sweep.py` | cos across checkpoints and sizes; permutation/rotation controls; OV; gated-model gate structure | weights only, minutes per model |
| 2 | `02_activations.py` | firing rate, gradient exposure, radial projection on in-distribution text; cross-lagged partials and unit trajectories across checkpoints | 2 M tokens/checkpoint, ~1 min each on H100 |
| 3 | `03_ablation.py` | remove / flip / keep-only aligned component vs norm-matched random; radial test; per-layer | minutes |
| 4 | `04_intervene.py` | **preregistered** continued-pretraining interventions: `component:<α>` (c→f) and `bias:<β>:random` (f→c), multi-seed | ~1.5 min / 1000 steps for 70m at 32×2048 tokens/step |
| 5 | `05_analyze_interventions.py` | arm-vs-matched-control comparison across seeds, dose-response fits, growth law, markdown report | seconds |
| 6 | `06_toy.py` | CPU toy transformer reproducing Phase 1 sign prediction and both intervention arms | ~4 min per 1500-step run on one core |

### Suggested H100 run plan (≈ one working day)

```bash
# Stage 1: does the phenomenon and its two timescales hold across sizes? (CPU ok, ~1 h total)
python scripts/01_checkpoint_sweep.py --models EleutherAI/pythia-70m EleutherAI/pythia-160m \
    EleutherAI/pythia-410m EleutherAI/pythia-1b --checkpoints coarse
# gated family with checkpoints (sign question): OLMo-2 1B revisions are listed on its HF model card
python scripts/01_checkpoint_sweep.py --models allenai/OLMo-2-0425-1B --revisions stage1-step10000-tokens21B stage1-step100000-tokens210B main

# Stage 2: activation-space mechanism on real text (GPU, ~10 min)
python scripts/02_activations.py --model EleutherAI/pythia-70m --revisions step8000 step32000 step143000 --device cuda
python scripts/02_activations.py --model EleutherAI/pythia-410m --revisions step8000 step32000 step143000 --device cuda

# Stage 3: load-bearing + radial (GPU, ~5 min per model)
python scripts/03_ablation.py --model EleutherAI/pythia-70m --device cuda
python scripts/03_ablation.py --model EleutherAI/pythia-410m --device cuda

# Stage 4: preregistered interventions (GPU; 5 seeds x 6 arms x 8k steps ≈ 6 h for 70m)
for s in 0 1 2 3 4; do
  for arm in control component:-1 component:2 bias:0.5:random bias:1:random bias:2:random; do
    python scripts/04_intervene.py --model EleutherAI/pythia-70m --revision step8000 --arm $arm --seed $s --bf16
  done
done
python scripts/05_analyze_interventions.py --model EleutherAI/pythia-70m --revision step8000
```

Decision rules for each stage are in `docs/PREREGISTRATION.md`. Read it before Stage 4 and do not
change it after looking at Stage-4 results.

## Conventions that matter

* `read` is `(m, d)` with **rows** = read vectors `a_k`; `write` is `(d, m)` with **columns** = write vectors
  `d_k`; `cos_k = cos(a_k, d_k)`. For gated MLPs the read vector is `up_proj`'s row; the gate is reported
  separately. Firing is `P(pre-activation > 0)` of the GeLU input (or of the gate for gated MLPs).
* Gradient exposure `E[h_k (g·â_k)]` (with `g = ∂L/∂(mlp_out)`) is the mechanistic variable behind
  "firing"; the dose-response analysis in the sandbox showed realized `P(z>0)` is *not* a sufficient
  statistic (imposed bias predicted Δc better), so Stage 4 records both.
* Matched runs: an arm and its control with the same `--seed` see identical batches in identical order.
  Never compare arms across seeds.
* Hooks never return values (a hook returning a tensor replaces the module output; this bug produced a
  loss of 8.8 instead of 4.0 during development — `test_hooks_do_not_alter_model_output` guards it).
  The one intentional output modification, `VirtualBias`, wraps `forward` so that every observer sees it.

## Known limitations / things to check first

* `adapters.py` supports `gpt_neox` and llama-like families (`llama, mistral, qwen2/3, olmo, olmo2, gemma`).
  GPT-2's `Conv1D` layout is not wired up (raises `NotImplementedError`).
* Pythia's `step8000` learning rate at that point of its schedule is ~1e-3 (cosine from 1e-3 for 70m);
  `04_intervene.py` defaults to `--lr 3e-4`. Continued pretraining at 32×2048 tokens/step is 1/32 of
  Pythia's 2 M-token batches, so 8000 steps here ≈ 250 Pythia steps of tokens. The control arm reports
  whether the alignment drifts under these settings; if it drifts a lot, raise the batch or lower the lr.
* The OV "copy alignment" grew monotonically to the end of training in the sandbox, contrary to the
  induction-head plateau consensus; Stage 1 measures it at every checkpoint so this can be checked
  rather than assumed.
* The gated-model *sign* profile (positive early, negative late in SmolLM2; negative early in Pythia)
  is documented in Gerstner & Schütze but unexplained. Stage 1 on OLMo checkpoints plus the `q`
  measurement in `06_toy.py --style smol` is the test of the two-factor prediction for that case.
