# ppc-forecaster

[![PyPI](https://img.shields.io/pypi/v/ppc-forecaster)](https://pypi.org/project/ppc-forecaster/)
[![CI](https://github.com/PPSwipUp/ppc-forecaster/actions/workflows/CI.yml/badge.svg)](https://github.com/PPSwipUp/ppc-forecaster/actions/workflows/CI.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PPSwipUp/ppc-forecaster/blob/main/notebooks/quickstart.ipynb)

A time-series forecaster that **keeps learning while it runs**. Point it at a CSV or a live feed and pick
what to predict. It forecasts every row before seeing the answer, learns from each answer as it arrives,
and tells you honestly whether it beats simple baselines on *your* data.

Under the hood is a plastic recurrent network written in Rust (the "PPC brain"): leaky neurons at many
timescales, Hebbian fast weights, and recursive-least-squares readouts. A plain online linear model and
naive baselines run beside it, and `auto` mode uses whichever has been most accurate recently.

## Install

```bash
pip install ppc-forecaster
```

Prebuilt wheels are published for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon) and Windows
(x64), for Python 3.9 and later. On other platforms pip builds from source, which needs a Rust toolchain.

**Try it in your browser:** the [quickstart notebook](notebooks/quickstart.ipynb) (click the Colab badge) forecasts
London's weather and improves a professional weather forecast, in about two minutes.

## Examples

| script | what it shows |
|---|---|
| [`examples/weather_bias_correction.py`](examples/weather_bias_correction.py) | improve a weather model's 24 h forecast for any lat/lon by learning its local errors |
| [`examples/home_energy.py`](examples/home_energy.py) | a household's electricity use, 1 h and 1 day ahead |
| [`examples/website_traffic.py`](examples/website_traffic.py) | daily Wikipedia page views; shows `auto` falling back to "no change" when the learners lose |

Run them from the repository root: `python examples/home_energy.py` (they download public data on first run).

## Command line

```bash
# stream a file through every model, print an honest score table, save the model
ppc train weather.csv --target temperature --time time --horizons 1,24 --season 24,168 --exog all --save temp.ppc

# later: feed new rows, get forecasts, keep learning
ppc update temp.ppc new_rows.csv --out forecasts.csv

# improve an existing forecast: column `nwp24` holds, on row t, someone else's forecast for t+24
ppc train weather.csv --target wind --time time --horizons 24 --base 24:nwp24
```

## Python

```python
import pandas as pd
from ppc.forecaster import Forecaster

df = pd.read_csv("weather.csv")
f = Forecaster(horizons=(1, 24), season=[24, 168])      # daily + weekly cycles
pred = f.fit_predict(df, target="temperature", time="time", exog=["pressure", "wind"])
print(f.report())          # MAE / RMSE, skill vs persistence, interval coverage
f.save("temp.ppc")

f = Forecaster.load("temp.ppc")
new_pred = f.update(new_rows)   # continues exactly where it stopped
```

Output columns: `<model>_h<H>` is that model's forecast, made on that row, for the value H rows later.
The models are `brain`, `linear`, `persistence`, `seasonal` (plus `seasonal<s>` for extra seasons), `base`
(if given) and `auto`. `auto_h<H>_lo` / `_hi` give an 80% prediction interval (`interval=0.8`; `None` turns it off).

**Useful options**
- `season=[24, 168]` sets several cycles (in rows).
- `transform="auto"` (the default) learns on a log scale for skewed, non-negative data like counts or sales;
  `"log"` or `None` force it.
- `base={24: "col"}` gives it an existing forecast to improve.

## How it is scored

Everything is **prequential**: the forecast for row t+H is made at row t, and the models learn from it
only when row t+H arrives. There is no train/test split to leak, so the score table is out-of-sample.
Skill = 1 − MAE / MAE(persistence), where persistence means "no change".

## Benchmarks (honest)

Average absolute error on held-out data (lower is better), `season=[24, 168]`, no extra inputs:

| data | ahead | brain | linear | no change |
|---|---|---|---|---|
| London temperature (°C), 2016–2025 | 1 h | **0.388** | 0.393 | 0.653 |
| | next 24 h | 1.460 | **1.368** | 1.747 |
| Bike-share rentals per hour (UCI) | 1 h | **32.2** | 34.0 | 79.8 |
| | next 24 h | 63.9 | **60.6** | 170.7 |
| Household power, kW (UCI) | 1 h | **0.361** | 0.365 | 0.414 |
| London wind, correcting a professional weather forecast | 24 h | 1.95 | **1.84** | 4.63 (professional forecast alone: 1.99) |

- **The brain shines at short horizons** where inputs interact (bike demand by hour and working day).
- **At 24 h+ a linear model usually wins.** `auto` picks per horizon for you.
- **From history alone it will not beat a physics-based weather model**, but given that forecast with
  `--base` it learns its local biases online.

Reproduce with `python benchmarks/compare.py` and `python -m experiments.forecast_bench` from the source repository.

**Against other libraries** ([full results](benchmarks/RESULTS.md)). Univariate, River tuned, ppc on defaults:
- **1 hour ahead:** ppc is best on all three datasets, 7–31% below the best River model.
- **24 hours ahead:** ppc beats statsforecast MSTL on temperature (−12%) and household power (−11%) and ties it
  on bike rentals.
- **Its 80% prediction intervals held 79–80% of actual values.**

## Limits

- It is a forecaster, not a trading system. Tested on financial markets and sports betting, it found no
  edge that survives costs, and nothing here claims otherwise.
- Numeric columns only. Missing inputs get a missing-flag rather than a fake value.

## Development

```bash
pip install maturin pytest
maturin develop --release          # builds the Rust engine into your environment
pytest tests/test_forecaster.py
```

`ppc_rs/build.sh` builds a copy tuned for your own CPU (faster, not portable; never ship it).
Releasing is described in `RELEASING.md`.

## License

MIT
