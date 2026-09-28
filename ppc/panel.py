"""PanelForecaster: many series at once (every shop, sensor, city...), one continual learner per series.

    from ppc.panel import PanelForecaster
    pf = PanelForecaster(id_col="store", horizons=(1, 24), season=[24, 168], fast=True)
    pred = pf.fit_predict(df, target="sales", time="time")     # df has all stores stacked, any order
    pf.report()                    # per model: mean skill across series, and how many series it won
    pf.report(by_series=True)      # the full table for every series
    pred_new = pf.update(new_rows) # rows for known stores continue their model; new stores get a new one

Each series gets its own Forecaster (same settings), so a series only learns from its own history; the
series run in parallel threads (the Rust engine releases the GIL).  Nothing is pooled across series.
"""
from __future__ import annotations

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .forecaster import Forecaster


class PanelForecaster:
    def __init__(self, id_col, n_jobs=None, **forecaster_kwargs):
        self.id_col = id_col
        self.n_jobs = n_jobs or os.cpu_count() or 4
        self.kw = forecaster_kwargs
        self.models = {}
        self.target = self.time = None
        self.exog = []
        probe = Forecaster(**forecaster_kwargs)                 # validates the settings once
        self.horizons, self.interval = probe.horizons, probe.interval

    def _split(self, df):
        if self.id_col not in df:
            raise KeyError(f"id column '{self.id_col}' not in {list(df.columns)}")
        out = {}
        for k, g in df.groupby(self.id_col, sort=False):
            out[k] = g.sort_values(self.time, kind="stable") if self.time else g
        return out

    def _map(self, fn, items):
        with ThreadPoolExecutor(min(self.n_jobs, max(1, len(items)))) as ex:
            return list(ex.map(fn, items))

    def _fit_one(self, item):
        k, g = item
        f = Forecaster(**self.kw)
        P = f.fit_predict(g.drop(columns=[self.id_col]), target=self.target, time=self.time, exog=self.exog)
        return k, f, P

    def fit_predict(self, df, target, time=None, exog=None):
        self.target, self.time = target, time
        self.exog = [c for c in (exog or []) if c not in (target, time, self.id_col)]
        self.models = {}
        out = []
        for k, f, P in self._map(self._fit_one, list(self._split(df).items())):
            self.models[k] = f
            out.append(P.assign(**{self.id_col: k}))
        return pd.concat(out, ignore_index=True)

    def update(self, df_new):
        """new rows for known series continue their model; unseen series start a new one"""
        if self.target is None:
            raise RuntimeError("call fit_predict first (or load a saved model)")

        def one(item):
            k, g = item
            if k in self.models:
                return k, self.models[k], self.models[k].update(g.drop(columns=[self.id_col]))
            return self._fit_one(item)

        out = []
        for k, f, P in self._map(one, list(self._split(df_new).items())):
            self.models[k] = f
            out.append(P.assign(**{self.id_col: k}))
        return pd.concat(out, ignore_index=True)

    def report(self, by_series=False, warmup=0.1):
        """by_series=False: one row per horizon x model, averaged over series (skill is scale-free, so it is
        the fair average; MAE is averaged too but big series dominate it), plus `wins` = number of series where
        that model had the lowest error in the second half.  by_series=True: every series' own report."""
        tabs = []
        for k, f in self.models.items():
            R = f.report(warmup=warmup)
            R.insert(0, self.id_col, k)
            tabs.append(R)
        R = pd.concat(tabs, ignore_index=True)
        if by_series:
            return R
        best = R[R.model != "auto"].loc[lambda d: d.groupby([self.id_col, "horizon"]).MAE_2nd_half.idxmin()]
        wins = best.groupby(["horizon", "model"]).size()
        cols = [c for c in R.columns if c not in (self.id_col, "horizon", "model")]
        A = R.groupby(["horizon", "model"], sort=False)[cols].mean().reset_index()
        A["series"] = R.groupby(["horizon", "model"], sort=False).size().values
        A["wins"] = [int(wins.get((h, m), 0)) for h, m in zip(A.horizon, A.model)]
        return A

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
