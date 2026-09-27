"""Prequential (predict-then-learn) runner and scoring.

Every prediction is recorded at the tick it is made, before its answer exists, so the
whole log is out-of-sample by construction.  Scoring aligns predictions with answers
afterwards; answers are never fed back to the brain from here.
"""
from __future__ import annotations

import time

import numpy as np

from .heads import GVFHead, PredictHead


def run(brain, stream, sleep_every=None, sleep_frac=0.5, progress=0):
    """stream yields (x_t, signals_t).  Returns dict head -> (preds[T,dim], signals[T,dim])."""
    preds = {k: [] for k in brain.heads}
    sigs = {k: [] for k in brain.heads}
    t0 = time.time()
    for t, (x, signals) in enumerate(stream):
        out = brain.step(x, signals)
        for k, head in brain.heads.items():
            preds[k].append(np.atleast_1d(out[k]))
            s = signals.get(k) if signals else None
            sigs[k].append(np.full(head.dim, np.nan) if s is None else np.atleast_1d(np.asarray(s, float)))
        if sleep_every and (t + 1) % sleep_every == 0:
            brain.sleep(sleep_frac)
        if progress and (t + 1) % progress == 0:
            print(f"  t={t + 1}  {(t + 1) / (time.time() - t0):.0f} ticks/s  {brain.stats()}")
    return {k: (np.array(preds[k]), np.array(sigs[k])) for k in brain.heads}


def targets_for(head, sig):
    """What each prediction should be scored against (NaN where unknown)."""
    T = len(sig)
    y = np.full_like(sig, np.nan)
    if isinstance(head, PredictHead):
        d = head.delay
        y[:T - d] = sig[d:]
    elif isinstance(head, GVFHead):
        # realised discounted future cumulant G_t = sum_{k>=1} gamma^(k-1) c_{t+k}
        g, G = head.gamma, np.zeros(sig.shape[1])
        c = np.nan_to_num(sig)
        for t in range(T - 2, -1, -1):
            G = c[t + 1] + g * G
            y[t] = G
        y[max(0, T - int(5 * head.horizon)):] = np.nan   # truncated tail is not a fair target
    return y


def score(pred, y, skip=0):
    """Out-of-sample metrics over ticks >= skip where the target exists."""
    p, y = pred[skip:].ravel(), y[skip:].ravel()
    m = np.isfinite(y) & np.isfinite(p)
    p, y = p[m], y[m]
    if len(y) < 10:
        return {"n": int(len(y))}
    mse = float(np.mean((y - p) ** 2))
    var = float(np.var(y))
    corr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else 0.0
    nz = y != 0
    hit = float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else float("nan")
    return {"n": int(len(y)), "mse": mse, "r2": 1 - mse / var if var > 0 else float("nan"),
            "corr": corr, "hit": hit}


def evaluate(brain, log, skip=0, windows=None):
    """Per-head scores; optional windows=[(start,end,label)] for per-segment scores."""
    res = {}
    for k, head in brain.heads.items():
        p, s = log[k]
        y = targets_for(head, s)
        res[k] = {"all": score(p, y, skip)}
        for (a, b, lab) in (windows or []):
            res[k][lab] = score(p[a:b], y[a:b])
    return res
