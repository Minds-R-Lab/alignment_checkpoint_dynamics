"""Statistics used across experiments. Pure numpy/scipy; no model access."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, rankdata


def partial_spearman(x, y, z) -> float:
    """Spearman corr(x, y | z): rank-transform, regress out z linearly, correlate residuals."""
    x, y, z = [rankdata(np.asarray(v, float)) for v in (x, y, z)]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def cross_lagged(c_early, f_early, c_late, f_late) -> dict:
    """The two directional tests: does early geometry predict later firing beyond early firing, and
    does early firing predict later geometry beyond early geometry?"""
    return dict(
        rho_f_c_early=float(spearmanr(f_early, c_early)[0]),
        rho_f_c_late=float(spearmanr(f_late, c_late)[0]),
        geom_to_fire=partial_spearman(c_early, f_late, f_early),
        fire_to_geom=partial_spearman(f_early, c_late, c_early))


def trajectory_stability(c_early, c_late, thr=-0.5, top_frac=0.2) -> dict:
    tail = c_late < thr
    lead = c_early < np.percentile(c_early, 100 * top_frac)
    return dict(corr=float(np.corrcoef(c_early, c_late)[0, 1]),
                tail_from_leaders=float(np.mean(lead[tail])) if tail.any() else np.nan,
                leaders_to_tail=float(np.mean(tail[lead])),
                leaders_churned=float(np.mean(np.abs(c_late[lead]) < 0.1)))


def dose_response(df_realized, dc, c0) -> dict:
    """Fits of the geometric response to the realized firing change."""
    one = np.ones_like(dc)
    def fit(X):
        b = np.linalg.lstsq(X, dc, rcond=None)[0]
        r2 = 1 - ((dc - X @ b) ** 2).sum() / ((dc - dc.mean()) ** 2).sum()
        return b, float(r2)
    b1, r1 = fit(np.stack([df_realized, one], 1))
    b2, r2 = fit(np.stack([df_realized, df_realized ** 2, c0 * df_realized, one], 1))
    return dict(linear_slope=float(b1[0]), linear_r2=r1, nonlin=b2[:3].tolist(), nonlin_r2=r2,
                spearman=float(spearmanr(df_realized, dc)[0]))


def forward_arm(dc, df, c0) -> dict:
    """Within-layer forward arm c -> f, measured on ONE layer of a single-layer intervention.

    dc, df are arm-minus-control changes in cos and firing for the units of the intervened layer; c0 is
    their pre-intervention cos. The forward-arm hypothesis (H7) is that, within a layer, making a unit
    LESS anti-aligned (dc > 0) lowers its firing (df < 0), i.e. Spearman(dc, df) < 0. `partial_given_c0`
    removes the static firing-alignment relationship so only the intervention-INDUCED coupling remains.
    `slope` is the OLS df ~ dc coefficient (sign-comparable to a derivative dν/dc)."""
    dc, df, c0 = np.asarray(dc, float), np.asarray(df, float), np.asarray(c0, float)
    ok = np.isfinite(dc) & np.isfinite(df) & np.isfinite(c0)
    dc, df, c0 = dc[ok], df[ok], c0[ok]
    slope = float(np.polyfit(dc, df, 1)[0]) if dc.std() > 0 else float("nan")
    return dict(spearman=float(spearmanr(dc, df)[0]), partial_given_c0=partial_spearman(dc, df, c0),
                slope=slope, dc_mean=float(dc.mean()), df_mean=float(df.mean()), n=int(dc.size))


def growth_law(f0, c0, dc) -> dict:
    """dc_i = a (f_i - mean f) + b c_i + const on a control run."""
    X = np.stack([f0 - f0.mean(), c0, np.ones_like(f0)], 1)
    b = np.linalg.lstsq(X, dc, rcond=None)[0]
    r2 = 1 - ((dc - X @ b) ** 2).sum() / ((dc - dc.mean()) ** 2).sum()
    return dict(a=float(b[0]), b=float(b[1]), r2=float(r2), partial_f_dc_given_c=partial_spearman(f0, dc, c0))


def md_table(rows: list[dict], cols: list[str], fmt="{:+.3f}") -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return head + body
