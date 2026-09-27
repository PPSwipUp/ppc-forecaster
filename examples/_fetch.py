"""Tiny download helpers shared by the examples (standard library + pandas only)."""
import io
import json
import ssl
import urllib.request
import zipfile

import pandas as pd

UA = {"User-Agent": "ppc-forecaster-examples/0.1 (https://github.com/PPSwipUp/ppc-forecaster)"}


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300, context=_ctx()).read()


def weather_with_forecast(lat, lon, start="2024-02-05", end=None):
    """Hourly observed-ish weather plus the forecast a professional weather model issued ~1 day earlier
    (Open-Meteo previous-runs API, free, no key).  Column `nwp24_<var>` on row t = forecast for t+24."""
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    hv = ("temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover,"
          "temperature_2m_previous_day1,wind_speed_10m_previous_day1")
    url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={end}&hourly={hv}")
    w = pd.DataFrame(json.loads(get(url))["hourly"])
    w["nwp24_temperature"] = w.temperature_2m_previous_day1.shift(-24)
    w["nwp24_wind"] = w.wind_speed_10m_previous_day1.shift(-24)
    return w.drop(columns=["temperature_2m_previous_day1", "wind_speed_10m_previous_day1"])


def household_power():
    """UCI 'Individual household electric power consumption' (2006-2010), hourly mean kW."""
    raw = get("https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip")
    z = zipfile.ZipFile(io.BytesIO(raw))
    p = pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".txt")][0]), sep=";", na_values="?",
                    usecols=["Date", "Time", "Global_active_power"], low_memory=False)
    p["time"] = pd.to_datetime(p.Date + " " + p.Time, dayfirst=True)
    return p.set_index("time").Global_active_power.resample("1h").mean().rename("kw").reset_index()


def wikipedia_views(article="Python_(programming_language)", start="20160101", end=None):
    """Daily human page views of an English Wikipedia article (Wikimedia REST API, free)."""
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).strftime("%Y%m%d")
    a = urllib.request.quote(article, safe="")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
           f"{a}/daily/{start}/{end}")
    items = json.loads(get(url))["items"]
    return pd.DataFrame({"time": pd.to_datetime([i["timestamp"][:8] for i in items]),
                         "views": [i["views"] for i in items]})
