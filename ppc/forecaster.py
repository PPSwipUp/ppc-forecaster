"""Forecaster: point the PPC brain at a time series and it keeps learning as rows arrive.

    from ppc.forecaster import Forecaster
    f = Forecaster(horizons=(1, 24), season=24)
    pred = f.fit_predict(df, target="temp", time="time", exog=["pressure", "wind"])   # every row, prequential
    print(f.report())                        # honest scores vs persistence / seasonal-naive / linear
    f.save("model.ppc")
    ...
    f = Forecaster.load("model.ppc")
    pred_new = f.update(new_rows)            # continues from where it stopped, keeps learning

Every forecast is made before the value it predicts is seen ("prequential"): the forecast of y[t+h] is
made at row t and the model learns from it only once row t+h arrives.  So the scores are out-of-sample
by construction, with no train/test split to leak.

Models run side by side on the same features:
  brain        PPC recurrent network (Rust engine), RLS readout
  linear       the same readout with no network (online recursive least squares on the features)
  persistence  y[t+h] = y[t]
  seasonal     y[t+h] = y[t+h-season]            (only if season >= h)
  base         an existing forecast you supply (base={h: column}, the column's row t = its forecast for t+h)
  auto         at each row, whichever of the above has the lowest recent error (known errors only)
The brain and linear models predict the correction to a starting forecast: the supplied base forecast if
given (so they learn its biases online, "model output statistics"), else persistence y[t].
"""
from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .brain import Brain
from .heads import PredictHead

MODELS = ("brain", "linear")


def _lagdiff(y, k):
    out = np.full(len(y), np.nan)
    out[k:] = y[k:] - y[:-k]
    return out


def _roll(y, w, fn):
    s = pd.Series(y).rolling(w, min_periods=1)
    return getattr(s, fn)().values


class Forecaster:
    def __init__(self, horizons=(1,), season=None, lags=None, windows=None, n_neurons=128, seed=0,
                 auto_halflife=500, rls_halflife=200_000.0, base=None):
        self.horizons = tuple(int(h) for h in horizons)
        self.season = season
        self.lags = tuple(lags) if lags else tuple(sorted({1, 2, 3, 6, 12, *( [season, 2 * season] if season else [24])}))
        self.windows = tuple(windows) if windows else (6, 24) if not season else (max(2, season // 4), season)
        self.n_neurons, self.seed = n_neurons, seed
        self.base = {int(h): c for h, c in (base or {}).items()}
        self.auto_halflife, self.rls_halflife = auto_halflife, rls_halflife
        self.brains = None
        self.target = self.time = None
        self.exog = []
        self.miss = {}
        self.hist = None                         # tail of past rows needed to continue
        self.ewma = None                         # recent |error| per horizon/model for "auto"
        self.pending = []                        # forecasts made but not yet scored (for auto after resume)
        self.log = None                          # full prequential log (for report)

    # ---------------------------------------------------------------- features
    def _features(self, y, t, E):
        cols = [y]
        cols += [_lagdiff(y, k) for k in self.lags]
        for w in self.windows:
            cols += [_roll(y, w, "mean") - y, _roll(y, w, "std")]
        if t is not None:
            ts = pd.DatetimeIndex(t)
            for per, v in ((24, ts.hour + ts.minute / 60), (7, ts.dayofweek), (365.25, ts.dayofyear)):
                cols += [np.sin(2 * np.pi * v / per), np.cos(2 * np.pi * v / per)]
        for j in range(E.shape[1]):
            cols += [E[:, j], _lagdiff(E[:, j], 1)]            # (flag columns just get a harmless diff)
        return np.nan_to_num(np.column_stack(cols).astype(float))

    def _start(self, y, B, h):
        """starting forecast for y[t+h] made at t: the supplied base forecast (if finite) else y[t]"""
        if h in self.base:
            b = B[self.base[h]]
            return np.where(np.isfinite(b), b, y)
        return y

    def _labels(self, y, B):
        out = {}
        for h in self.horizons:
            s0 = self._start(y, B, h)
            lab = np.full(len(y), np.nan)
            lab[h:] = y[h:] - s0[:-h]              # at row t: truth minus the start made h rows ago
            out[f"d{h}"] = lab[:, None]
        return out

    def _new_brains(self, D):
        out = {}
        for m in MODELS:
            heads = [PredictHead(f"d{h}", delay=h, lr=0.01, solver="rls", rls_halflife=self.rls_halflife)
                     for h in self.horizons]
            n = self.n_neurons if m == "brain" else 0
            from .rust import RustBrain
            out[m] = RustBrain.from_numpy(Brain(D, heads, n_neurons=n, seed=self.seed), seed=self.seed + 1)
        return out

    def _prep(self, df):
        y = pd.to_numeric(df[self.target], errors="coerce").ffill().values.astype(float)
        t = pd.to_datetime(df[self.time]).values if self.time else None
        if not self.exog:
            return y, t, np.zeros((len(df), 0))
        raw = df[self.exog].apply(pd.to_numeric, errors="coerce")
        E = raw.ffill()
        # an input that is missing (e.g. before a data source started) gets a 0/1 flag instead of a fake 0 value
        flags = [E[c].isna().values.astype(float) for c in self.exog if self.miss.get(c)]
        E = E.fillna(0).values.astype(float)
        return y, t, np.column_stack([E, *flags]) if flags else E

    # ---------------------------------------------------------------- run
    def _run(self, df_all, n_new):
        """df_all = kept history + new rows; learn/predict on the last n_new rows"""
        y, t, E = self._prep(df_all)
        B = {c: pd.to_numeric(df_all[c], errors="coerce").values.astype(float) for c in self.base.values()}
        X = self._features(y, t, E)
        for h, c in self.base.items():             # the correction model sees the base forecast relative to y[t]
            X = np.column_stack([X, np.nan_to_num(B[c] - y), np.isfinite(B[c]).astype(float)])
        lab = self._labels(y, B)
        new = slice(len(y) - n_new, len(y))
        if self.brains is None:
            self.brains = self._new_brains(X.shape[1])
        with ThreadPoolExecutor(len(MODELS)) as ex:          # Rust releases the GIL: models run in parallel
            outs = dict(zip(MODELS, ex.map(lambda m: self.brains[m].run(X[new], {k: v[new] for k, v in lab.items()}),
                                           MODELS)))
        yn = y[new]
        res = {"y": yn}
        if t is not None:
            res["time"] = t[new]
        for h in self.horizons:
            s0 = self._start(y, B, h)[new]
            for m in MODELS:
                res[f"{m}_h{h}"] = s0 + outs[m][f"d{h}"][:, 0]
            res[f"persistence_h{h}"] = yn
            if h in self.base:
                res[f"base_h{h}"] = B[self.base[h]][new]
            if self.season and self.season >= h:
                idx = np.arange(len(y))[new] + h - self.season
                res[f"seasonal_h{h}"] = np.where(idx >= 0, y[np.maximum(idx, 0)], np.nan)
        P = pd.DataFrame(res)
        self._auto(P, y, new)
        keep = max(max(self.lags), max(self.windows), max(self.horizons), self.season or 0) + 2
        self.hist = df_all.iloc[-keep:].copy()
        return P

    def _cands(self, h):
        return [m for m in (*MODELS, "persistence", "seasonal", "base")
                if (m != "seasonal" or (self.season and self.season >= h)) and (m != "base" or h in self.base)]

    def _auto(self, P, y, new):
        """auto_h: at each row use the model with the lowest EWMA |error| among errors already known"""
        a = 0.5 ** (1 / self.auto_halflife)
        if self.ewma is None:
            self.ewma = {h: {m: np.nan for m in self._cands(h)} for h in self.horizons}
            self.pending = []
        # pending forecasts from an earlier call, plus this call's, resolved in time order
        rows = len(P)
        for h in self.horizons:
            cands = self._cands(h)
            F = np.column_stack([P[f"{m}_h{h}"].values for m in cands])
            prev = [p for p in self.pending if p[0] == h]
            choice = np.empty(rows)
            for i in range(rows):
                # resolve forecasts whose target is row i (made at row i-h)
                j = i - h
                if j >= 0:
                    f = F[j]
                elif prev and len(prev[0][1]) >= -j:
                    f = prev[0][1][j]
                else:
                    f = None
                if f is not None:
                    for k, m in enumerate(cands):
                        e = abs(f[k] - y[new][i])
                        if np.isfinite(e):
                            o = self.ewma[h][m]
                            self.ewma[h][m] = e if np.isnan(o) else a * o + (1 - a) * e
                scores = [self.ewma[h][m] for m in cands]
                k = int(np.nanargmin(scores)) if np.isfinite(scores).any() else cands.index("persistence")
                choice[i] = F[i, k]
            P[f"auto_h{h}"] = choice
        self.pending = [(h, np.column_stack([P[f"{m}_h{h}"].values for m in self._cands(h)])[-h:])
                        for h in self.horizons]

    def fit_predict(self, df, target, time=None, exog=None):
        self.target, self.time = target, time
        self.exog = [c for c in (exog or []) if c != target and c != time]
        self.miss = {c: bool(pd.to_numeric(df[c], errors="coerce").isna().any()) for c in self.exog}
        P = self._run(df.reset_index(drop=True), len(df))
        self.log = P
        return P

    def update(self, df_new):
        """continue learning on new rows (same columns as fit_predict) and return their forecasts"""
        if self.hist is None:
            raise RuntimeError("call fit_predict first (or load a saved model)")
        df_all = pd.concat([self.hist, df_new], ignore_index=True)
        P = self._run(df_all, len(df_new))
        self.log = P if self.log is None else pd.concat([self.log, P], ignore_index=True)
        return P

    # ---------------------------------------------------------------- scoring
    def report(self, P=None, warmup=0.1):
        """MAE / RMSE per horizon and model, scored against the value h rows later; skill = 1 - MAE/MAE(persistence).
        Rows in the first `warmup` fraction are skipped (every model is still learning there); the
        second-half columns show whether the ranking holds up later."""
        P = self.log if P is None else P
        y = P.y.values
        n = len(y)
        rows = []
        for h in self.horizons:
            truth = np.full(n, np.nan)
            truth[:n - h] = y[h:]
            for m in self._cands(h) + ["auto"]:
                f = P[f"{m}_h{h}"].values
                r = {"horizon": h, "model": m}
                for part, lo in (("all", int(warmup * n)), ("2nd_half", n // 2)):
                    e = (f - truth)[lo:]
                    e = e[np.isfinite(e)]
                    r[f"MAE_{part}"] = np.abs(e).mean()
                    r[f"RMSE_{part}"] = np.sqrt((e ** 2).mean())
                rows.append(r)
        R = pd.DataFrame(rows)
        for part in ("all", "2nd_half"):
            base = R[R.model == "persistence"].set_index("horizon")[f"MAE_{part}"]
            R[f"skill_{part}"] = 1 - R[f"MAE_{part}"] / R.horizon.map(base)
        return R

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
