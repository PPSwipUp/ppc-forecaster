"""Forecast hourly temperature for ten cities at once with PanelForecaster (one learner per city).

  python examples/many_cities.py
"""
import json

import pandas as pd

from _fetch import get
from ppc.panel import PanelForecaster

CITIES = {"London": (51.51, -0.13), "Paris": (48.85, 2.35), "Madrid": (40.42, -3.70), "Berlin": (52.52, 13.40),
          "Rome": (41.90, 12.50), "Oslo": (59.91, 10.75), "New York": (40.71, -74.01), "Tokyo": (35.68, 139.69),
          "Sydney": (-33.87, 151.21), "Cape Town": (-33.92, 18.42)}

parts = []
for city, (lat, lon) in CITIES.items():
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date=2023-01-01&end_date=2025-12-31&hourly=temperature_2m,relative_humidity_2m,pressure_msl")
    parts.append(pd.DataFrame(json.loads(get(url))["hourly"]).assign(city=city))
df = pd.concat(parts, ignore_index=True)
print(f"{len(df)} rows, {df.city.nunique()} cities\n")

pf = PanelForecaster("city", horizons=(1, 6, 24), season=24, fast=True)
pf.fit_predict(df, target="temperature_2m", time="time", exog=["relative_humidity_2m", "pressure_msl"])

A = pf.report()
print(A[["horizon", "model", "skill_2nd_half", "wins", "coverage_2nd_half"]].round(3).to_string(index=False, na_rep=""))
print("\nskill = 1 - error / error of 'no change' (averaged over cities); wins = cities where that model was best")
R = pf.report(by_series=True)
print("\n1-hour-ahead error by city (°C):")
print(R[(R.horizon == 1) & (R.model == "auto")].set_index("city").MAE_2nd_half.round(3).to_string())
