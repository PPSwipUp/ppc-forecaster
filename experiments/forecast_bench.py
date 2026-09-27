"""Benchmark the Forecaster (ppc/forecaster.py) on real public datasets, all prequential.

  weather_era5   London hourly 2016-2025 (Open-Meteo archive = ERA5 reanalysis): temperature, wind
  weather_nwp    London hourly 2022-2026 (Open-Meteo previous-runs API): the professional weather-model
                 forecast issued ~1 day ahead ("previous_day1") is the benchmark for the 24 h forecast;
                 the "+nwp" run also gives the forecaster that NWP forecast as an input (online bias
                 correction, what weather companies call MOS)
  power          UCI household electric power, hourly mean kW, 2006-2010
  bike           UCI bike sharing, hourly rentals, 2011-2012, with the current weather as inputs

  python -m experiments.forecast_bench        (downloads ~25 MB on first run into ./bench_data)
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppc.forecaster import Forecaster  # noqa: E402

# data/results location: $PPC_BENCH_DIR, else ./bench_data
_ROOT = os.environ.get("PPC_BENCH_DIR", os.path.join(os.getcwd(), "bench_data"))
D = os.path.join(_ROOT, "data")
OUT = os.path.join(_ROOT, "runs")


def _ssl_ctx():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
LAT, LON = 51.51, -0.13
VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover,precipitation"


def _json(url):
    return json.loads(urllib.request.urlopen(url, timeout=120, context=_ssl_ctx()).read())


def fetch():
    os.makedirs(D, exist_ok=True)
    fp = os.path.join(D, "weather_era5.csv")
    if not os.path.exists(fp):
        parts = []
        for y in range(2016, 2026):
            j = _json(f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
                      f"&start_date={y}-01-01&end_date={y}-12-31&hourly={VARS}")
            parts.append(pd.DataFrame(j["hourly"]))
        pd.concat(parts).to_csv(fp, index=False)
    fp = os.path.join(D, "weather_nwp.csv")
    if not os.path.exists(fp):
        parts = []
        hv = VARS + ",temperature_2m_previous_day1,wind_speed_10m_previous_day1"
        for a, b in (("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
                     ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-09-25")):
            j = _json(f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
                      f"&start_date={a}&end_date={b}&hourly={hv}")
            parts.append(pd.DataFrame(j["hourly"]))
        w = pd.concat(parts).reset_index(drop=True)
        # the NWP forecast FOR t+24, issued about a day before t+24 (so known at t): shift back 24 rows
        w["nwp24_temp"] = w.temperature_2m_previous_day1.shift(-24)
        w["nwp24_wind"] = w.wind_speed_10m_previous_day1.shift(-24)
        w.to_csv(fp, index=False)
    fp = os.path.join(D, "power.csv")
    if not os.path.exists(fp):
        raw = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
                                     timeout=300, context=_ssl_ctx()).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        p = pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".txt")][0]), sep=";", na_values="?",
                        usecols=["Date", "Time", "Global_active_power"], low_memory=False)
        p["time"] = pd.to_datetime(p.Date + " " + p.Time, dayfirst=True)
        h = p.set_index("time").Global_active_power.resample("1h").mean().rename("kw").reset_index()
        h.to_csv(fp, index=False)
    fp = os.path.join(D, "bike.csv")
    if not os.path.exists(fp):
        raw = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
                                     timeout=300, context=_ssl_ctx()).read()
        b = pd.read_csv(zipfile.ZipFile(io.BytesIO(raw)).open("hour.csv"))
        b["time"] = pd.to_datetime(b.dteday) + pd.to_timedelta(b.hr, unit="h")
        b[["time", "cnt", "temp", "atemp", "hum", "windspeed", "weathersit", "holiday", "workingday"]].to_csv(fp, index=False)


JOBS = [
    ("weather_era5", "temperature_2m", (1, 6, 24), ["relative_humidity_2m", "pressure_msl", "wind_speed_10m", "cloud_cover", "precipitation"]),
    ("weather_era5", "wind_speed_10m", (1, 6, 24), ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover", "precipitation"]),
    ("weather_nwp", "temperature_2m", (24,), ["relative_humidity_2m", "pressure_msl", "wind_speed_10m", "cloud_cover", "precipitation"]),
    ("weather_nwp", "temperature_2m", (24,), ["relative_humidity_2m", "pressure_msl", "wind_speed_10m", "cloud_cover", "precipitation", "nwp24_temp"]),
    ("weather_nwp", "wind_speed_10m", (24,), ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover", "precipitation"]),
    ("weather_nwp", "wind_speed_10m", (24,), ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover", "precipitation", "nwp24_wind"]),
    ("power", "kw", (1, 24), []),
    ("bike", "cnt", (1, 24), ["temp", "atemp", "hum", "windspeed", "weathersit", "holiday", "workingday"]),
]


def run_job(job):
    name, target, hs, exog = job
    df = pd.read_csv(os.path.join(D, f"{name}.csv"))
    if name == "weather_nwp":                    # the NWP forecast archive starts 2024-02-04: score only those rows
        df = df[df.nwp24_temp.notna() | (df.time > "2024-02-04")].reset_index(drop=True)
    nwp = [e for e in exog if e.startswith("nwp24")]
    f = Forecaster(horizons=hs, season=24, base={24: nwp[0]} if nwp else None)
    exog = [e for e in exog if not e.startswith("nwp24")]
    P = f.fit_predict(df, target=target, time="time", exog=exog)
    R = f.report()
    tag = f"{name}:{target}" + ("+nwp" if nwp else "")
    R.insert(0, "dataset", tag)
    extra = None
    if name == "weather_nwp":
        # the professional forecast's own score on the same rows/half the report uses (value for t+24 vs truth)
        nwp = df["nwp24_temp" if target == "temperature_2m" else "nwp24_wind"].values
        y = df[target].values
        n = len(y)
        truth = np.full(n, np.nan)
        truth[:n - 24] = y[24:]
        e = (nwp - truth)
        extra = {"dataset": tag, "horizon": 24, "model": "NWP (professional)"}
        for part, lo in (("all", int(0.1 * n)), ("2nd_half", n // 2)):
            ee = e[lo:][np.isfinite(e[lo:])]
            extra[f"MAE_{part}"] = np.abs(ee).mean()
            extra[f"RMSE_{part}"] = np.sqrt((ee ** 2).mean())
        base = R[R.model == "persistence"].iloc[0]
        for part in ("all", "2nd_half"):
            extra[f"skill_{part}"] = 1 - extra[f"MAE_{part}"] / base[f"MAE_{part}"]
    out = pd.concat([R, pd.DataFrame([extra])]) if extra else R
    return out


def main():
    fetch()
    with ProcessPoolExecutor(min(len(JOBS), os.cpu_count())) as ex:
        res = pd.concat(list(ex.map(run_job, JOBS)), ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "bench.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res[["dataset", "horizon", "model", "MAE_all", "skill_all", "MAE_2nd_half", "skill_2nd_half"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
