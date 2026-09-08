"""Tests run on tiny randomly-initialized models (no downloads). Each test names the bug it catches."""
import numpy as np, torch, pytest
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM
from rwloop import metrics as M
from rwloop.adapters import get_layers, firing_module
from rwloop.hooks import collect, Recorder
from rwloop.intervene import ablate, snapshot, restore, scale_aligned, shift_bias, VirtualBias
from rwloop.analysis import partial_spearman, cross_lagged, dose_response
from rwloop.train import continue_pretraining

rng = np.random.default_rng(0)


def tiny_neox():
    cfg = GPTNeoXConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                        max_position_embeddings=64, rotary_pct=0.25, use_parallel_residual=True)
    torch.manual_seed(0); return GPTNeoXForCausalLM(cfg).eval()


def tiny_llama():
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0); return LlamaForCausalLM(cfg).eval()


# ------------------------------------------------------------------ metrics
def test_unit_cos_is_per_unit_and_sign_correct():
    """Catches: transposed write matrix or wrong axis. Plant d_k = -2 a_k for k<5, random otherwise."""
    A = rng.standard_normal((20, 8)); W = rng.standard_normal((8, 20))
    W[:, :5] = -2 * A[:5].T
    c = M.unit_cos(A, W)
    assert np.allclose(c[:5], -1.0, atol=1e-12)
    assert np.all(np.abs(c[5:]) < 0.9)
    lam = M.aligned_coeffs(A, W)
    assert np.allclose(lam[:5], -2.0, atol=1e-12)


def test_aligned_energy_fraction_bounds():
    A = rng.standard_normal((30, 10)); W = -A.T.copy()          # fully aligned -> 1.0
    assert abs(M.aligned_energy_fraction(A, W) - 1.0) < 1e-12
    R = rng.standard_normal((10, 30)); R -= (np.sum(R.T * (A / np.linalg.norm(A, axis=1, keepdims=True)), 1, keepdims=True) * (A / np.linalg.norm(A, axis=1, keepdims=True))).T
    assert M.aligned_energy_fraction(A, R) < 1e-10              # orthogonal residual -> 0


def test_permutation_control_kills_alignment_but_rotation_keeps_mean():
    """Catches: rotation applied to only one factor (would change the composed map / kill the mean)."""
    # polarized population: half the units strongly self-adjoint, half not (the Phase-2 signature)
    A = rng.standard_normal((400, 64)); lam = np.where(np.arange(400) < 200, -1.5, 0.0)
    W = lam[None, :] * A.T + 0.8 * rng.standard_normal((64, 400))
    c = M.unit_cos(A, W); assert c.mean() < -0.3 and c.std() > 0.3
    assert abs(M.permutation_control(A, W, rng).mean()) < 0.05
    Q, _ = np.linalg.qr(rng.standard_normal((400, 400)))
    Ar, Wr = Q @ A, W @ Q.T
    B = W @ A
    assert np.allclose(Wr @ Ar, B, atol=1e-8)                              # composed map exactly unchanged
    # the EXACT invariant is the sum of per-unit dot products = tr(B); the mean cosine is only approximate
    assert abs(np.sum(Ar * Wr.T) - np.trace(B)) < 1e-8 and abs(np.sum(A * W.T) - np.trace(B)) < 1e-8
    rot = M.unit_cos(Ar, Wr)
    assert abs(rot.mean() - c.mean()) < 0.2                                # approximately preserved
    assert rot.std() < c.std() / 2 and np.mean(rot < -0.8) < 0.01 < np.mean(c < -0.8)   # concentration destroyed


def test_spectral_surrogate_preserves_singular_values():
    X = rng.standard_normal((30, 50)) * np.linspace(1, 5, 30)[:, None]
    S = M.spectral_surrogate(X, rng)
    assert np.allclose(np.linalg.svd(X, compute_uv=False), np.linalg.svd(S, compute_uv=False), atol=1e-8)
    assert np.linalg.norm(S - X) > 1.0     # and it is actually a different matrix


def test_block_trace_ratio_matches_definition():
    A = rng.standard_normal((16, 8)); W = rng.standard_normal((8, 16))
    B = W @ A
    assert abs(M.block_trace_ratio(A, W) - np.trace(B) / np.linalg.norm(B)) < 1e-12


# ------------------------------------------------------------------ adapters
@pytest.mark.parametrize("make", [tiny_neox, tiny_llama])
def test_adapter_shapes_and_conventions(make):
    m = make(); layers = get_layers(m)
    assert len(layers) == 3
    for L in layers:
        R, W = L.read(), L.write()
        assert R.shape == (64, 32) and W.shape == (32, 64)          # rows=read vectors, cols=write vectors
        assert L.v().shape == (32, 32) and L.o().shape == (32, 32)
        c = M.unit_cos(R, W); assert c.shape == (64,)
    # packed qkv slicing: planting a value into v rows must show up in L.v()
    L = layers[0]
    if L.packed_qkv:
        with torch.no_grad():
            W = L.v_mod.weight.view(4, 3, 8, 32); W[:, 2] = 7.0
        assert torch.all(L.v() == 7.0)


# ------------------------------------------------------------------ hooks
def test_hooks_do_not_alter_model_output():
    """REGRESSION: a forward hook returning a tensor replaces the module output. This bug once fed a
    frequency counter into the MLP and produced loss 8.8 instead of 4.0."""
    m = tiny_neox(); layers = get_layers(m); x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        ref = m(input_ids=x).logits.clone()
    rec = Recorder(layers, want_grad=False).attach()
    with torch.no_grad():
        out = m(input_ids=x).logits
    rec.detach()
    assert torch.allclose(ref, out)
    with torch.no_grad():
        again = m(input_ids=x).logits
    assert torch.allclose(ref, again)           # hooks fully removed


def test_firing_and_exposure_shapes_and_ranges():
    m = tiny_neox(); layers = get_layers(m); batches = [torch.randint(0, 100, (2, 16)) for _ in range(3)]
    res, loss = collect(m, layers, batches, "cpu", want_grad=True)
    assert loss > 0
    for i in range(3):
        f = res["firing"][i]; e = res["exposure"][i]
        assert f.shape == (64,) and np.all(f >= 0) and np.all(f <= 1) and 0 < f.mean() < 1
        assert e.shape == (64,) and np.isfinite(e).all() and np.abs(e).sum() > 0
    assert len(res["radial"]) == 3 and all(np.isfinite(r) for r in res["radial"])
    assert all(n > 0 for n in res["resid_norm"])


def test_exposure_matches_manual_computation():
    """Catches: wrong contraction in the exposure formula. Compare against an explicit loop on one batch."""
    m = tiny_neox(); layers = get_layers(m); x = torch.randint(0, 100, (1, 8))
    L = layers[1]
    store = {}
    h1 = L.write_mod.register_forward_pre_hook(lambda mod, a: store.__setitem__("h", a[0].detach().reshape(-1, 64)))
    h2 = L.mlp.register_full_backward_hook(lambda mod, gi, go: store.__setitem__("g", go[0].detach().reshape(-1, 32)))
    m.zero_grad(); m(input_ids=x, labels=x).loss.backward(); h1.remove(); h2.remove(); m.zero_grad()
    A = L.read(); Ah = A / A.norm(dim=1, keepdim=True)
    manual = torch.stack([(store["h"][:, k] * (store["g"] @ Ah[k])).sum() for k in range(64)]) / 8
    res, _ = collect(m, layers, [x], "cpu", want_grad=True)
    assert torch.allclose(torch.tensor(res["exposure"][1]).float(), manual.float(), atol=1e-5)


# ------------------------------------------------------------------ interventions
def test_remove_aligned_zeroes_cos_and_restore_roundtrips():
    m = tiny_neox(); layers = get_layers(m)
    with torch.no_grad():   # plant strong alignment so the test is not vacuous
        L = layers[1]; L.write_mod.weight.copy_(-0.5 * L.read_mod.weight.T + 0.005 * torch.randn(32, 64))
    snap = snapshot(layers)
    c0 = M.unit_cos(layers[1].read(), layers[1].write()); assert c0.mean() < -0.5
    ablate([layers[1]], "remove_aligned")
    assert np.all(np.abs(M.unit_cos(layers[1].read(), layers[1].write())) < 1e-5)
    restore(layers, snap)
    assert np.allclose(M.unit_cos(layers[1].read(), layers[1].write()), c0, atol=1e-6)
    ablate([layers[1]], "flip_aligned")
    assert np.allclose(M.unit_cos(layers[1].read(), layers[1].write()), -c0, atol=1e-5); restore(layers, snap)
    ablate([layers[1]], "keep_only_aligned")
    assert np.allclose(np.abs(M.unit_cos(layers[1].read(), layers[1].write())), 1.0, atol=1e-5); restore(layers, snap)


def test_remove_random_is_norm_matched_and_orthogonal():
    m = tiny_neox(); layers = get_layers(m); L = layers[1]
    with torch.no_grad():
        L.write_mod.weight.copy_(-0.5 * L.read_mod.weight.T + 0.005 * torch.randn(32, 64))
    D0 = L.write().T.clone(); Ah = L.read() / L.read().norm(dim=1, keepdim=True)
    proj_norm = ((D0 * Ah).sum(1, keepdim=True) * Ah).norm(dim=1)
    ablate([L], "remove_random", torch.Generator().manual_seed(0))
    delta = D0 - L.write().T
    assert torch.allclose(delta.norm(dim=1), proj_norm, atol=1e-5)          # same norm per unit
    assert torch.all((delta * Ah).sum(1).abs() < 1e-5)                      # removed direction ⟂ read
    assert np.allclose(M.unit_cos(L.read(), L.write()) * L.write().T.norm(dim=1).numpy(),
                       (D0 * Ah).sum(1).numpy(), atol=1e-4)                  # aligned component untouched


def test_scale_aligned_changes_cos_in_expected_direction():
    m = tiny_neox(); layers = get_layers(m); L = layers[2]
    with torch.no_grad():
        L.write_mod.weight.copy_(-0.3 * L.read_mod.weight.T + 0.01 * torch.randn(32, 64))
    c0 = M.unit_cos(L.read(), L.write())
    scale_aligned([L], 2.0); c2 = M.unit_cos(L.read(), L.write())
    # tripling the aligned component strengthens each unit's alignment IN ITS OWN SIGN
    assert np.all(np.sign(c2) == np.sign(c0)) and np.all(np.abs(c2) > np.abs(c0)) and c2.mean() < c0.mean() < 0
    scale_aligned([L], -1.0); assert np.all(np.abs(M.unit_cos(L.read(), L.write())) < 1e-5)


def test_virtual_bias_shifts_preactivation_and_groups_are_disjoint():
    m = tiny_neox(); layers = get_layers(m); x = torch.randint(0, 100, (2, 16))
    store = {}
    hk = firing_module(layers[1]).register_forward_hook(lambda mod, i, o: store.__setitem__("pre", o.detach().clone()))
    with torch.no_grad(): m(input_ids=x)
    pre0 = store["pre"]
    vbs, groups = shift_bias(m, [layers[1]], beta_std=1.0, frac=0.25, seed=0, batch=x, device="cpu")
    with torch.no_grad(): m(input_ids=x)
    pre1 = store["pre"]; hk.remove()
    g = groups[1]; assert (g == 1).sum() == 16 and (g == -1).sum() == 16 and ((g == 1) & (g == -1)).sum() == 0
    d = (pre1 - pre0)[0, 0]
    assert torch.all(d[g == 1] > 0) and torch.all(d[g == -1] < 0) and torch.all(d[g == 0].abs() < 1e-6)
    assert vbs[0].b.requires_grad
    vbs[0].detach()
    with torch.no_grad(): m(input_ids=x)
    with torch.no_grad(): m(input_ids=x)
    hk2 = firing_module(layers[1]).register_forward_hook(lambda mod, i, o: store.__setitem__("pre", o.detach().clone()))
    with torch.no_grad(): m(input_ids=x)
    hk2.remove(); assert torch.allclose(store["pre"], pre0)   # detach restored the original forward


# ------------------------------------------------------------------ analysis
def test_partial_spearman_removes_confound():
    z = rng.standard_normal(2000); x = z + 0.1 * rng.standard_normal(2000); y = z + 0.1 * rng.standard_normal(2000)
    assert abs(partial_spearman(x, y, z)) < 0.15          # x,y related only through z
    y2 = x + 0.03 * rng.standard_normal(2000)   # residual corr = 0.1/sqrt(0.01+0.0009) ~ 0.96
    assert partial_spearman(x, y2, z) > 0.8                # direct relation survives


def test_cross_lagged_direction_detects_planted_causality():
    """Plant: later firing depends on early geometry, not vice versa. The test must recover geom_to_fire >> fire_to_geom."""
    c0 = rng.standard_normal(3000); f0 = rng.standard_normal(3000)
    f1 = 0.5 * f0 - 0.8 * c0 + 0.3 * rng.standard_normal(3000)
    c1 = 0.9 * c0 + 0.3 * rng.standard_normal(3000)
    r = cross_lagged(c0, f0, c1, f1)
    assert r["geom_to_fire"] < -0.6 and abs(r["fire_to_geom"]) < 0.15


def test_dose_response_recovers_planted_slope():
    df = rng.uniform(-0.3, 0.6, 2000); c0 = rng.uniform(-0.6, 0.0, 2000)
    dc = -0.2 * df + 0.25 * df ** 2 - 0.2 * c0 * df + 0.005 * rng.standard_normal(2000)
    d = dose_response(df, dc, c0)
    assert np.allclose(d["nonlin"], [-0.2, 0.25, -0.2], atol=0.03) and d["nonlin_r2"] > 0.9


# ------------------------------------------------------------------ training loop
def test_continue_pretraining_snapshots_and_matched_order():
    """Two runs with the same seed and no intervention must be bit-identical; an on_step intervention must be visible."""
    def stream(seed):
        g = torch.Generator().manual_seed(seed)
        while True:
            yield torch.randint(0, 100, (2, 16), generator=g)
    def run(intervene):
        torch.manual_seed(1); m = tiny_neox(); layers = get_layers(m)
        def on_step(step, model, opt):
            if intervene and step == 3: scale_aligned(layers[1:], -1.0)
        snaps, losses = continue_pretraining(m, stream(0), 6, 1e-3, "cpu", [2, 6], window=2, bf16=False,
                                             on_step=on_step, log_every=100, want_exposure=True)
        return snaps, losses
    s1, l1 = run(False); s2, l2 = run(False); s3, l3 = run(True)
    assert np.allclose(l1, l2)
    assert np.allclose(s1[6]["mlp"][1], s2[6]["mlp"][1])
    assert set(s1) == {2, 6} and "firing" in s1[2] and "exposure" in s1[2] and s1[2]["firing"][0].shape == (64,)
    assert not np.allclose(s1[6]["mlp"][1], s3[6]["mlp"][1])     # intervention changed the trajectory
    assert np.allclose(l1[:3], l3[:3]) and not np.allclose(l1[3:], l3[3:])


def test_gated_gate_cosines_have_unit_shape():
    """REGRESSION: 01_checkpoint_sweep crashed on OLMo-2 because cos(gate, up) passed two (m,d) matrices."""
    m = tiny_llama(); L = get_layers(m)[0]; R, W, G = L.read(), L.write(), L.gate()
    assert M.unit_cos(G, W).shape == (64,) and M.unit_cos(G, R.T).shape == (64,)
    with pytest.raises(ValueError):
        M.unit_cos(G, R)          # the bug: broadcasting (64,32)*(32,64)
