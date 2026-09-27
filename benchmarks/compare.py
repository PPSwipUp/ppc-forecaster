"""Head-to-head: ppc-forecaster vs River (online) vs statsforecast (batch), univariate, honest.

Datasets (downloaded by experiments/forecast_bench.py): London hourly temperature 2016-2025, UCI household
power (hourly kW), UCI bike-share rentals (hourly).  Gaps are filled forward so every method sees the
same regular hourly series.  No exogenous inputs for anyone.

Task A  one step ahead, every hour (prequential: forecast, then learn).  Scored on the second half.
        ppc (brain / linear / auto), River SNARIMAX, River Holt-Winters, River linear regression on lags
        (River's settings tuned on the first half; ppc uses its defaults),
        naive (last value), seasonal naive (same hour yesterday).
        statsforecast is left out here: it re-fits a batch model, and doing that every hour is not practical.
Task B  day-ahead: at 00:00 each day forecast the next 24 hours, for the last DAYS days.
        ppc (24 horizons, streamed), River (forecast(24) from the online model),
        statsforecast AutoETS / MSTL(24, 168) / SeasonalNaive (AutoARIMA: too slow to re-fit daily), re-fitted every day on the last
        INPUT_DAYS days (rolling-origin evaluation, one fit per day, run in parallel).  Score = MAE over all 24 leads.

  PPC_BENCH_DIR=... python benchmarks/compare.py
"""
from __future__ import annotations

import os
import sys
import time

# statsforecast fits many models in parallel processes; stop each one from also starting its own maths
# threads (8 processes x 8 threads each thrashed the CPU)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

# import the installed ppc-forecaster (with its compiled engine) BEFORE experiments/, which puts the
# source folder on sys.path; appending the repo root keeps the installed package first
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppc.forecaster import Forecaster  # noqa: E402
from experiments.forecast_bench import D, OUT, fetch  # noqa: E402

DAYS = 100
INPUT_DAYS = 56
SETS = {"temperature (°C)": ("weather_era5", "temperature_2m"), "household power (kW)": ("power", "kw"),
        "bike rentals / h": ("bike", "cnt")}


def load(name, col):
    df = pd.read_csv(os.path.join(D, f"{name}.csv"), usecols=["time", col])
    df["time"] = pd.to_datetime(df.time)
    df = df.drop_duplicates("time").set_index("time").asfreq("h")
    df[col] = df[col].ffill()
    return df.reset_index().rename(columns={col: "y"}).dropna()


def mae(a, b):
    e = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return float(np.nanmean(e))


# ---------------------------------------------------------------- River (online)
def river_candidates():
    """River models and the settings tried for each; the best is picked on the FIRST half of every dataset
    (the half nobody is scored on), which is generous to River: ppc gets no tuning at all."""
    from river import compose, linear_model, optim, preprocessing, time_series
    sn = lambda lr: lambda: time_series.SNARIMAX(
        p=2, d=1, q=1, m=24, sp=1, sd=0, sq=1,
        regressor=compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression(optimizer=optim.SGD(lr))))
    hw = lambda a, g: lambda: time_series.HoltWinters(alpha=a, beta=0.01, gamma=g, seasonality=24)
    lin = lambda lr: lambda: compose.Pipeline(preprocessing.StandardScaler(),
                                              linear_model.LinearRegression(optimizer=optim.SGD(lr)))
    return {
        "River SNARIMAX": {f"lr={lr}": sn(lr) for lr in (0.01, 0.003, 0.001)},
        "River HoltWinters": {f"a={a},g={g}": hw(a, g) for a in (0.1, 0.3, 0.6) for g in (0.1, 0.3, 0.6)},
        "River linear on lags": {f"lr={lr}": lin(lr) for lr in (0.01, 0.005, 0.001)},
    }


def river_tune(y, name, configs):
    half = len(y) // 2
    best = None
    for cfg, make in configs.items():
        one, _ = river_run(y[:half], make, name, set(), half // 2)       # score on the 2nd quarter
        e = np.nanmean(np.abs(one[half // 2:-1] - y[half // 2 + 1:half]))
        if np.isfinite(e) and (best is None or e < best[0]):
            best = (e, cfg)
    return best[1]


LAGS = (1, 2, 3, 24, 168)


def river_run(y, make, name, origins, task_a_from):
    """one pass: one-step forecasts for every row (task A) + 24-step forecasts at the day-ahead origins (task B)"""
    m = make()
    one = np.full(len(y), np.nan)
    day = {}
    lagged = name.endswith("lags")
    for t in range(len(y)):
        # learn the value that just arrived (y[t]) first, then forecast y[t+1] / y[t+1..t+24]
        if lagged:
            if t >= 169:
                m.learn_one({f"l{k}": y[t - k] for k in LAGS}, y[t])          # lags known at t-1 -> target y[t]
            if t >= 168:
                x = {f"l{k}": y[t - k + 1] for k in LAGS}                    # lags known at t -> predict y[t+1]
                if t >= task_a_from:
                    one[t] = m.predict_one(x)
                if t in origins:
                    # no native multi-step for a lag regressor: roll it forward on its own forecasts
                    hist = list(y[t - 167:t + 1])
                    out = []
                    for _ in range(24):
                        v = m.predict_one({f"l{k}": hist[-k] for k in LAGS})
                        out.append(v)
                        hist.append(v)
                    day[t] = out
        else:
            m.learn_one(y[t])
            if t >= task_a_from:
                one[t] = m.forecast(horizon=1)[0]
            if t in origins:
                day[t] = list(m.forecast(horizon=24))
    return one, day


# ---------------------------------------------------------------- statsforecast (batch, re-fit daily)
def sf_day_ahead(df, mids):
    """rolling-origin day-ahead forecasts: one series per origin (its last INPUT_DAYS days), fitted in parallel.
    Same as StatsForecast.cross_validation(refit=True, input_size=...), but spread over all CPU cores."""
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoETS, SeasonalNaive
    y, ts = df.y.values.astype(float), df.time.values
    parts = [pd.DataFrame({"unique_id": i, "ds": ts[o - 24 * INPUT_DAYS + 1:o + 1], "y": y[o - 24 * INPUT_DAYS + 1:o + 1]})
             for i, o in enumerate(mids)]
    # AutoARIMA left out: 20-60 s per fit here, x300 daily re-fits is over an hour
    sf = StatsForecast(models=[AutoETS(season_length=24), MSTL(season_length=[24, 168]),
                               SeasonalNaive(season_length=24)], freq="h", n_jobs=-1)
    fc = sf.forecast(df=pd.concat(parts, ignore_index=True), h=24)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    return {m: np.stack([fc[fc.unique_id == i][m].values for i in range(len(mids))])
            for m in fc.columns if m not in ("unique_id", "ds")}


def main():
    fetch()
    rows = []
    for label, (name, col) in SETS.items():
        df = load(name, col)
        y = df.y.values.astype(float)
        T = len(y)
        half = T // 2
        mids = np.flatnonzero(df.time.dt.hour.values == 23)             # origin row = 23:00, forecast 00:00..23:00
        mids = mids[mids + 24 < T][-DAYS:]
        origins = set(mids.tolist())
        truth_day = np.stack([y[o + 1:o + 25] for o in mids])
        truth_one = np.r_[y[1:], np.nan]                                 # value one row later

        # ppc
        t0 = time.time()
        f = Forecaster(horizons=range(1, 25), season=24)
        P = f.fit_predict(df, target="y", time="time")
        for m in ("brain", "linear", "auto"):
            fc_day = np.stack([[P[f"{m}_h{h}"].values[o] for h in range(1, 25)] for o in mids])
            rows.append({"data": label, "method": f"ppc {m}", "one_step_MAE": mae(P[f"{m}_h1"].values[half:-1], truth_one[half:-1]),
                         "day_ahead_MAE": mae(fc_day, truth_day), "seconds": time.time() - t0})
        rows.append({"data": label, "method": "naive (last value)", "one_step_MAE": mae(y[half:-1], truth_one[half:-1]),
                     "day_ahead_MAE": mae(np.repeat(y[mids][:, None], 24, 1), truth_day)})
        rows.append({"data": label, "method": "seasonal naive (24 h)",
                     "one_step_MAE": mae(y[half - 23:-24], truth_one[half:-1]),
                     "day_ahead_MAE": mae(np.stack([y[o - 23:o + 1] for o in mids]), truth_day)})
        # River
        for rname, configs in river_candidates().items():
            cfg = river_tune(y, rname, configs)
            t0 = time.time()
            one, day = river_run(y, configs[cfg], rname, origins, half)
            fc_day = np.stack([day[o] for o in mids])
            rows.append({"data": label, "method": f"{rname} ({cfg})", "one_step_MAE": mae(one[half:-1], truth_one[half:-1]),
                         "day_ahead_MAE": mae(fc_day, truth_day), "seconds": time.time() - t0})
            print(f"  {label}: {rname} {cfg} done ({time.time() - t0:.0f}s)", flush=True)
        # statsforecast
        t0 = time.time()
        sf = sf_day_ahead(df, mids)
        for model, fc_day in sf.items():
            rows.append({"data": label, "method": f"statsforecast {model}", "one_step_MAE": np.nan,
                         "day_ahead_MAE": mae(fc_day, truth_day), "seconds": time.time() - t0})
        print(f"  {label}: statsforecast done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "compare.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
