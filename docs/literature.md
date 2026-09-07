# Literature position (sweep of 6 Sep 2026)

**Observational papers that establish the static phenomenon (cite as prior work):**
- Gerstner & Schütze, *Understanding Gated Neurons in Transformers from Their Input-Output Functionality*, arXiv:2505.17936 (2025). cos(w_in, w_out) across 12 gated models; enrichment early-middle, depletion late; gate ≈ orthogonal; |cos| bounded by ~0.8. Lists training dynamics as future work. Follow-up tool GLUScope (arXiv:2602.23826) does not address dynamics.
- Gurnee et al., *Universal Neurons in GPT2 Language Models*, arXiv:2401.12181 (2024). Fig. 9: high-frequency neurons have near-opposite input/output weights, mostly in the first quarter of depth.
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021); Elhage et al. (2022, SoLU footnote): negative IO cosine as deletion; negative/positive OV eigenvalues.

**Unclaimed as of the sweep (this repo's targets):** checkpoint dynamics of per-unit cos; the two-timescale mean/concentration split; spectral-surrogate / permutation / rotation controls; the linearized two-factor mechanism and the `q` sign predictor; causal interventions in either direction; the load-bearing radial-contraction result.

**Adjacent, to position against:**
- Energy Transformer (Hoover et al., NeurIPS 2023) and *Revisiting Transformer Layer Parameterization Through Causal Energy Minimization* (arXiv:2605.07588, 2026): weight tying `W_out = W_in^T` by design. This repo asks whether untied training spontaneously acquires a partial version.
- *Conservativeness of untied auto-encoders* (arXiv:1506.07643, 2015): product of untied encoder/decoder becomes increasingly symmetric during training — the historical precedent.
- *Where Pretraining writes and Alignment reads* (arXiv:2605.16600, 2026): weight-delta alignment with activation/prediction subspaces across Pythia checkpoints; same outer-product mechanism (updates = Σ δ_t a_t^T), different quantity. Supporting citation, not overlap.
- Two-factor alignment theory: Saxe et al. 2013; Ji & Telgarsky 2019; Du, Hu & Lee 2018 (balancedness); saddle-to-saddle (arXiv:2106.15933); Min et al. 2023 (early neuron alignment, small init).
- Induction/OV over training: Olsson et al. 2022 (phase change 2.5–5B tokens); Dual-Route Induction (arXiv:2504.03022). Note: the weight-space OV copy-alignment measured here grew monotonically in the sandbox, unlike the behavioural induction score.
