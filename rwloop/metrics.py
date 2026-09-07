"""Weight-space observables and controls. All inputs are torch tensors or numpy arrays;
outputs are numpy. Conventions: read (m, d) rows = read vectors; write (d, m) cols = write vectors.
"""
from __future__ import annotations
import numpy as np
import torch


def _np(x):
    return x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64)


def _unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


def unit_cos(read, write) -> np.ndarray:
    """cos(a_k, d_k) for every hidden unit k. read (m,d), write (d,m) -> (m,)"""
    R, W = _unit_rows(_np(read)), _unit_rows(_np(write).T)
    return np.sum(R * W, axis=1)


def aligned_coeffs(read, write) -> np.ndarray:
    """lambda_k such that d_k = lambda_k * a_k + r_k with r_k ⟂ a_k (no normalization of d)."""
    R, W = _np(read), _np(write).T
    return np.sum(R * W, axis=1) / np.maximum(np.sum(R * R, axis=1), 1e-12)


def aligned_energy_fraction(read, write) -> float:
    """Fraction of ||write||_F^2 that lies along the read directions (the 10-20% number)."""
    R, W = _unit_rows(_np(read)), _np(write).T
    proj = np.sum(W * R, axis=1, keepdims=True) * R
    return float(np.sum(proj ** 2) / np.sum(W ** 2))


def tail_fraction(c: np.ndarray, thr: float = -0.5) -> float:
    return float(np.mean(c < thr))


def block_trace_ratio(read, write) -> float:
    """tr(W_write W_read) / ||W_write W_read||_F  -- the spectral (rotation-invariant) part."""
    B = _np(write) @ _np(read)
    return float(np.trace(B) / np.linalg.norm(B))


def summary(read, write, thr_neg=-0.5, thr_pos=0.5) -> dict:
    c = unit_cos(read, write)
    return dict(mean=float(c.mean()), std=float(c.std()), p10=float(np.percentile(c, 10)),
                p50=float(np.percentile(c, 50)), p90=float(np.percentile(c, 90)),
                frac_lt=tail_fraction(c, thr_neg), frac_gt=float(np.mean(c > thr_pos)),
                frac_lt_02=float(np.mean(c < -0.2)), frac_gt_02=float(np.mean(c > 0.2)),
                aligned_energy=aligned_energy_fraction(read, write), trace_ratio=block_trace_ratio(read, write))


# ---------------------------------------------------------------- controls
def permutation_control(read, write, rng: np.random.Generator) -> np.ndarray:
    """Destroy the unit correspondence, keep both matrices: cos(a_k, d_{pi(k)})."""
    W = _np(write)
    return unit_cos(read, W[:, rng.permutation(W.shape[1])])


def rotation_control(read, write, rng: np.random.Generator) -> np.ndarray:
    """Rotate the hidden basis by an orthogonal Q: read -> Q read, write -> write Q^T.
    Leaves the composed map write@read exactly unchanged; the MEAN cos is ~invariant (it is a trace),
    the per-unit distribution collapses to the mean."""
    R, W = _np(read), _np(write)
    Q, _ = np.linalg.qr(rng.standard_normal((R.shape[0], R.shape[0])))
    return unit_cos(Q @ R, W @ Q.T)


def spectral_surrogate(M, rng: np.random.Generator) -> np.ndarray:
    """Same singular values as M, Haar-random singular vectors."""
    M = _np(M)
    s = np.linalg.svd(M, compute_uv=False)
    Q1, _ = np.linalg.qr(rng.standard_normal((M.shape[0], len(s))))
    Q2, _ = np.linalg.qr(rng.standard_normal((M.shape[1], len(s))))
    return (Q1 * s) @ Q2.T


def surrogate_zscore(read, write, stat, rng: np.random.Generator, n: int = 8) -> tuple[float, float, float]:
    """z-score of stat(read, write) against spectral surrogates of BOTH factors."""
    t = stat(read, write)
    S = [stat(spectral_surrogate(read, rng), spectral_surrogate(write, rng)) for _ in range(n)]
    return float(t), float(np.mean(S)), float((t - np.mean(S)) / (np.std(S) + 1e-12))


# ---------------------------------------------------------------- attention OV
def ov_unit_cos(v, o) -> np.ndarray:
    """cos(value-input row k, output-projection column k) per attention channel. v (H*dh, d), o (d, H*dh)."""
    return unit_cos(v, o)
