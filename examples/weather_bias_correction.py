"""Make a professional weather forecast better for YOUR spot by learning its local errors as you go.

A weather model's 1-day forecast is good but carries local biases (terrain, city heat, coastline).
Give ppc-forecaster that forecast as a --base and it learns the corrections online.

  python examples/weather_bias_correction.py [lat lon]        (default: London)
"""
import sys

from _fetch import weather_with_forecast
from ppc.forecaster import Forecaster

lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (51.51, -0.13)
df = weather_with_forecast(lat, lon)
print(f"{len(df)} hours from {df.time.iloc[0]} to {df.time.iloc[-1]}\n")
for target, base in (("temperature_2m", "nwp24_temperature"), ("wind_speed_10m", "nwp24_wind")):
    f = Forecaster(horizons=(24,), season=24, base={24: base})
    f.fit_predict(df, target=target, time="time",
                  exog=["relative_humidity_2m", "pressure_msl", "cloud_cover", "wind_speed_10m", "temperature_2m"])
    R = f.report().set_index("model")
    print(f"--- {target}, 24 h ahead (second half of the data) ---")
    print(R[["MAE_2nd_half", "skill_2nd_half"]].round(3).to_string())
    gain = 1 - R.loc["auto", "MAE_2nd_half"] / R.loc["base", "MAE_2nd_half"]
    print(f"auto vs the professional forecast alone: {100 * gain:+.1f}% lower error\n")
