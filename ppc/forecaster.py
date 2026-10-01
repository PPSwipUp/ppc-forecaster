"""Forecaster: point the PPC brain at a time series and it keeps learning as rows arrive.

    from ppc.forecaster import Forecaster
    f = Forecaster(horizons=(1, 24), season=[24, 168])          # daily + weekly cycles
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
  seasonal     y[t+h] = y[t+h-s] for the first season s >= h;  seasonal<s> for each further season
  base         an existing forecast you supply (base={h: column}, the column's row t = its forecast for t+h)
  auto         at each row, whichever of the above has the lowest recent error (known errors only)
The brain and linear models predict the correction to a starting forecast: the supplied base forecast if
given (so they learn its biases online, "model output statistics"), else persistence y[t].

transform="auto" learns on log(1 + y) when the target is non-negative and strongly skewed (counts, sales,
page views), decided from the first rows only; forecasts are always returned in the original units.
interval=0.8 adds auto_h<H>_lo / _hi: an 80% prediction interval from the recent errors of the auto
forecast (online conformal), with its real coverage shown by report().
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


def _shift(y, k):
    """value k rows earlier (k > 0) or later (k < 0), NaN where it does not exist"""
    out = np.full(len(y), np.nan)
    if k > 0:
        out[k:] = y[:-k]
    elif k < 0:
        out[:k] = y[-k:]
    else:
        out[:] = y
    return out


class Forecaster:
    def __init__(self, horizons=(1,), season=None, lags=None, windows=None, n_neurons=128, seed=0,
                 auto_halflife=500, rls_halflife=200_000.0, base=None, transform="auto", interval=0.8,
                 interval_window=1000, brain_max_horizon=None, fast=False):
        """fast=True: the brain (the slow part) only forecasts horizons up to 6 steps, where it wins;
        longer horizons use the linear model and baselines.  Same as brain_max_horizon=6."""
        self.horizons = tuple(int(h) for h in horizons)
        seasons = [] if season is None else [season] if np.isscalar(season) else list(season)
        self.seasons = sorted(int(s) for s in seasons)
        self.season = self.seasons[0] if self.seasons else None
        extra = [k for s in self.seasons for k in (s, 2 * s)] or [24]
        self.lags = tuple(lags) if lags else tuple(sorted({1, 2, 3, 6, 12, *extra}))
        self.windows = (tuple(windows) if windows else
                        (6, 24) if not self.seasons else tuple(sorted({max(2, self.seasons[0] // 4), *self.seasons})))
        self.n_neurons, self.seed = n_neurons, seed
        self.base = {int(h): c for h, c in (base or {}).items()}
        self.auto_halflife, self.rls_halflife = auto_halflife, rls_halflife
        assert transform in ("auto", "log", None, "none")
        self.transform = None if transform == "none" else transform
        self.use_log = False
        self.interval, self.interval_window = interval, interval_window
        self.brain_max_horizon = 6 if fast and brain_max_horizon is None else brain_max_horizon
        self.brains = None
        self.target = self.time = None
        self.exog = []
        self.miss = {}
        self.hist = None                         # tail of past rows needed to continue
        self.ewma = None                         # recent |error| per horizon/model for "auto"
        self.pending = []                        # forecasts made but not yet scored (for auto after resume)
        self.resid, self.pend_int = {}, {}       # interval state: recent residuals, unscored auto forecasts
        self.log = None                          # full prequential log (for report)

    def __setstate__(self, st):                  # models saved by 0.1.x lack the newer settings
        self.__dict__.update(st)
        s = self.__dict__
        s.setdefault("seasons", [s["season"]] if s.get("season") else [])
        s.setdefault("transform", None)
        s.setdefault("use_log", False)
        s.setdefault("interval", None)
        s.setdefault("interval_window", 1000)
        s.setdefault("resid", {})
        s.setdefault("pend_int", {})
        s.setdefault("brain_max_horizon", None)
        if s.get("ewma") is not None and "ewma_w" not in s:     # saved before 0.3.1: treat as fully warmed up
            s["ewma_w"] = {h: {m: float(np.isfinite(v)) for m, v in d.items()} for h, d in s["ewma"].items()}

    # ---------------------------------------------------------------- transform
    def _fwd(self, v):
        return np.log1p(np.maximum(v, 0.0)) if self.use_log else v

    def _inv(self, v):
        return np.expm1(v) if self.use_log else v

    def _decide_transform(self, y):
        if self.transform == "log":
            return True
        if self.transform != "auto":
            return False
        head = y[:max(200, len(y) // 10)]
        head = head[np.isfinite(head)]
        if len(head) < 50 or head.min() < 0 or head.std() == 0:
            return False
        skew = ((head - head.mean()) ** 3).mean() / head.std() ** 3
        return bool(skew > 1.0)

    # ---------------------------------------------------------------- features
    def _features(self, z, t, E):
        cols = [z]
        cols += [_lagdiff(z, k) for k in self.lags]
        for w in self.windows:
            cols += [_roll(z, w, "mean") - z, _roll(z, w, "std")]
        # the value one season before each target, relative to now: z[t+h-s] - z[t]
        for h in self.horizons:
            for s in self.seasons:
                if s >= h:
                    cols.append(_shift(z, s - h) - z)
        if t is not None:
            ts = pd.DatetimeIndex(t)
            for per, v in ((24, ts.hour + ts.minute / 60), (7, ts.dayofweek), (365.25, ts.dayofyear)):
                cols += [np.sin(2 * np.pi * v / per), np.cos(2 * np.pi * v / per)]
        for j in range(E.shape[1]):
            cols += [E[:, j], _lagdiff(E[:, j], 1)]            # (flag columns just get a harmless diff)
        return np.nan_to_num(np.column_stack(cols).astype(float))

    def _start(self, z, B, h):
        """starting forecast (model space) for y[t+h] made at t: the base forecast if finite, else z[t]"""
        if h in self.base:
            b = B[self.base[h]]
            return np.where(np.isfinite(b), b, z)
        return z

    def _labels(self, z, B):
        out = {}
        for h in self.horizons:
            s0 = self._start(z, B, h)
            lab = np.full(len(z), np.nan)
            lab[h:] = z[h:] - s0[:-h]              # at row t: truth minus the start made h rows ago
            out[f"d{h}"] = lab[:, None]
        return out

    def _model_horizons(self, m):
        if m == "brain" and self.brain_max_horizon is not None:
            return [h for h in self.horizons if h <= self.brain_max_horizon]
        return list(self.horizons)

    def _models(self):
        return [m for m in MODELS if self._model_horizons(m)]

    def _new_brains(self, D):
        from .rust import RustBrain
        out = {}
        for m in self._models():
            heads = [PredictHead(f"d{h}", delay=h, lr=0.01, solver="rls", rls_halflife=self.rls_halflife)
                     for h in self._model_horizons(m)]
            n = self.n_neurons if m == "brain" else 0
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
    def _seasonal_names(self, h):
        return [("seasonal" if i == 0 else f"seasonal{s}", s) for i, s in enumerate(self.seasons) if s >= h]

    def _run(self, df_all, n_new):
        """df_all = kept history + new rows; learn/predict on the last n_new rows"""
        y, t, E = self._prep(df_all)
        z = self._fwd(y)
        B = {c: self._fwd(pd.to_numeric(df_all[c], errors="coerce").values.astype(float)) for c in self.base.values()}
        X = self._features(z, t, E)
        for h, c in self.base.items():             # the correction model sees the base forecast relative to now
            X = np.column_stack([X, np.nan_to_num(B[c] - z), np.isfinite(B[c]).astype(float)])
        lab = self._labels(z, B)
        new = slice(len(y) - n_new, len(y))
        if self.brains is None:
            self.brains = self._new_brains(X.shape[1])
        models = list(self.brains)
        with ThreadPoolExecutor(len(models)) as ex:          # Rust releases the GIL: models run in parallel
            outs = dict(zip(models, ex.map(lambda m: self.brains[m].run(
                X[new], {k: v[new] for k, v in lab.items() if k in self.brains[m].heads}), models)))
        yn = y[new]
        res = {"y": yn}
        if t is not None:
            res["time"] = t[new]
        for h in self.horizons:
            s0 = self._start(z, B, h)[new]
            for m in outs:
                if f"d{h}" in outs[m]:
                    res[f"{m}_h{h}"] = self._inv(s0 + outs[m][f"d{h}"][:, 0])
            res[f"persistence_h{h}"] = yn
            if h in self.base:
                res[f"base_h{h}"] = self._inv(B[self.base[h]][new])
            for name, s in self._seasonal_names(h):
                res[f"{name}_h{h}"] = _shift(y, s - h)[new]
        P = pd.DataFrame(res)
        self._auto(P, y, new)
        if self.interval:
            self._intervals(P, yn)
        keep = max(max(self.lags), max(self.windows), max(self.horizons), max(self.seasons, default=0)) + 2
        self.hist = df_all.iloc[-keep:].copy()
        return P

    def _cands(self, h):
        out = [m for m in MODELS if h in self._model_horizons(m)] + ["persistence"]
        out += [n for n, _ in self._seasonal_names(h)]
        return out + (["base"] if h in self.base else [])

    def _auto(self, P, y, new):
        """auto_h: at each row use the model with the lowest EWMA |error| among errors already known.
        The forecast for row i was made at row i-h, so its error is known at row i (vectorised, exact).
        The average is bias-corrected (weighted sum / sum of weights), so the first errors don't dominate it
        for hundreds of rows on short series."""
        alpha = 1 - 0.5 ** (1 / self.auto_halflife)
        if self.ewma is None:
            self.ewma = {h: {m: np.nan for m in self._cands(h)} for h in self.horizons}
            self.ewma_w = {h: {m: 0.0 for m in self._cands(h)} for h in self.horizons}
            self.pending = []
        prev_all = dict(self.pending)
        yn = y[new]
        n = len(P)
        pend = []
        for h in self.horizons:
            cands = self._cands(h)
            F = np.column_stack([P[f"{m}_h{h}"].values for m in cands])
            prev = prev_all.get(h, np.zeros((0, len(cands))))
            prev = np.vstack([np.full((h - len(prev), len(cands)), np.nan), prev]) if len(prev) < h else prev[-h:]
            Fx = np.vstack([prev, F])                                    # Fx[i] = forecasts for new row i
            err = np.abs(Fx[:n] - yn[:, None])
            # carried state: weighted error sum (as an average) and the weight it represents, starting at 0 / 0
            w0 = np.array([[self.ewma_w[h].get(m, 0.0) for m in cands]])
            s0 = np.nan_to_num(np.array([[self.ewma[h].get(m, np.nan) for m in cands]])) * w0
            ew = lambda X: pd.DataFrame(X).ewm(alpha=alpha, adjust=False, ignore_na=True).mean().values[1:]
            Sx = ew(np.vstack([s0, err]))                                # sum(weights * error)
            Wx = ew(np.vstack([w0, np.where(np.isfinite(err), 1.0, np.nan)]))  # sum(weights)
            with np.errstate(invalid="ignore", divide="ignore"):
                E = np.where(Wx > 0, Sx / Wx, np.nan)
            k = np.where(np.isfinite(E).any(1), np.argmin(np.where(np.isfinite(E), E, np.inf), 1),
                         cands.index("persistence"))
            P[f"auto_h{h}"] = F[np.arange(n), k]
            if n:
                self.ewma[h] = {m: E[-1, j] for j, m in enumerate(cands)}
                self.ewma_w[h] = {m: Wx[-1, j] for j, m in enumerate(cands)}
            pend.append((h, Fx[-h:]))
        self.pending = pend

    def _intervals(self, P, yn):
        """auto_h<H>_lo/_hi from the empirical quantiles of the auto forecast's recent errors (y - forecast),
        using only errors already known at each row (the forecast for row r was made h rows earlier)."""
        n, W = len(P), self.interval_window
        qlo, qhi = (1 - self.interval) / 2, (1 + self.interval) / 2
        for h in self.horizons:
            f = P[f"auto_h{h}"].values
            prev = self.pend_int.get(h, np.zeros(0))
            prev = np.concatenate([np.full(h - len(prev), np.nan), prev])          # forecasts for rows 0..h-1
            fx = np.concatenate([prev, f])                                         # fx[r] = forecast for new row r
            e_new = yn - fx[:n]                                                    # realised at row r
            e_all = np.concatenate([self.resid.get(h, np.zeros(0)), e_new])
            r = pd.Series(e_all).rolling(W, min_periods=30)
            lo, hi = r.quantile(qlo).values[-n:], r.quantile(qhi).values[-n:]
            P[f"auto_h{h}_lo"], P[f"auto_h{h}_hi"] = f + lo, f + hi
            self.resid[h] = e_all[-W:]
            self.pend_int[h] = fx[-h:]

    def fit_predict(self, df, target, time=None, exog=None):
        self.target, self.time = target, time
        self.exog = [c for c in (exog or []) if c != target and c != time]
        self.miss = {c: bool(pd.to_numeric(df[c], errors="coerce").isna().any()) for c in self.exog}
        self.use_log = self._decide_transform(pd.to_numeric(df[target], errors="coerce").values.astype(float))
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
        second-half columns show whether the ranking holds up later.  For `auto`, coverage_2nd_half is how
        often the truth actually fell inside the prediction interval (should be close to `interval`)."""
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
                if m == "auto" and f"auto_h{h}_lo" in P:
                    lo, hi = P[f"auto_h{h}_lo"].values[n // 2:], P[f"auto_h{h}_hi"].values[n // 2:]
                    tt = truth[n // 2:]
                    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(tt)
                    r["coverage_2nd_half"] = ((tt[ok] >= lo[ok]) & (tt[ok] <= hi[ok])).mean()
                    r["width_2nd_half"] = (hi[ok] - lo[ok]).mean()
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
