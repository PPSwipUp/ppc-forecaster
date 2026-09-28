"""Home-Assistant-free logic of the integration (so it can be tested without Home Assistant).

Home Assistant keeps raw state history only ~10 days, but keeps hourly *long-term statistics* for sensors
that have a state_class forever.  We train on those: `mean` for things like temperature or power (W),
`change` for meters that only count up (energy kWh, water, gas) -> usage per hour.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ppc.forecaster import Forecaster


def stats_to_frame(rows, mode):
    """rows: Home Assistant statistics rows (dicts with 'start' plus 'mean'/'change'/'sum'), one per hour.
    Returns a regular hourly DataFrame(time, y) in UTC; gaps are left NaN (the forecaster fills them)."""
    if not rows:
        return pd.DataFrame({"time": pd.to_datetime([], utc=True), "y": []})
    start = [r["start"] for r in rows]
    t = pd.to_datetime(start, unit="s", utc=True) if isinstance(start[0], (int, float)) else pd.to_datetime(start, utc=True)
    if mode == "change":
        v = [r.get("change") for r in rows]
        if all(x is None for x in v):                        # older HA: derive the change from the running sum
            s = pd.Series([r.get("sum") for r in rows], dtype=float)
            v = s.diff().values
    else:
        v = [r.get("mean") for r in rows]
    df = pd.DataFrame({"time": t, "y": pd.to_numeric(pd.Series(v), errors="coerce").values})
    df = df.drop_duplicates("time").set_index("time").sort_index()
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC"))
    return df.rename_axis("time").reset_index()


class Runner:
    """Owns one Forecaster for one sensor: first fit on history, then one update per new hour."""

    def __init__(self, horizons, seasons, fast=True):
        self.horizons, self.seasons, self.fast = tuple(horizons), list(seasons), fast
        self.model = None
        self.last_time = None

    def _clean(self, df):
        return df.assign(time=df.time.dt.tz_convert("UTC").dt.tz_localize(None), y=df.y)

    def fit(self, df):
        if df.y.notna().any():
            df = df.loc[df.y.first_valid_index():]              # start at the first hour with data
        if df.y.notna().sum() < 3 * max(self.seasons or [24]):
            raise ValueError(f"not enough history yet: {int(df.y.notna().sum())} hours "
                             f"(need at least {3 * max(self.seasons or [24])})")
        self.model = Forecaster(horizons=self.horizons, season=self.seasons or None, fast=self.fast)
        P = self.model.fit_predict(self._clean(df), target="y", time="time")
        self.last_time = df.time.iloc[-1]
        return P

    def update(self, df):
        """df: hourly rows (any overlap with what was seen is dropped); returns the new rows' forecasts or None"""
        new = df[df.time > self.last_time]
        if new.empty:
            return None
        P = self.model.update(self._clean(new))
        self.last_time = new.time.iloc[-1]
        return P

    def latest(self):
        """{h: dict(value, lower, upper, time_of_forecast, skill)} from the most recent row"""
        P = self.model.log
        R = self.model.report().set_index(["horizon", "model"])
        out = {}
        last = P.iloc[-1]
        t_last = pd.Timestamp(last["time"]).tz_localize(timezone.utc) if "time" in P else datetime.now(timezone.utc)
        for h in self.horizons:
            v = last.get(f"auto_h{h}")
            out[h] = {"value": None if v is None or not np.isfinite(v) else float(v),
                      "lower": _f(last.get(f"auto_h{h}_lo")), "upper": _f(last.get(f"auto_h{h}_hi")),
                      "for_time": (t_last + pd.Timedelta(hours=h)).isoformat(),
                      "skill_vs_no_change": _f(R.loc[(h, "auto"), "skill_2nd_half"]) if (h, "auto") in R.index else None,
                      "interval_coverage": _f(R.loc[(h, "auto"), "coverage_2nd_half"])
                      if (h, "auto") in R.index and "coverage_2nd_half" in R else None}
        return out


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
