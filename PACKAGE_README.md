# ppc-forecaster

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

## Command line

```bash
# stream a file through every model, print an honest score table, save the model
ppc train weather.csv --target temperature --time time --horizons 1,24 --season 24 --exog all --save temp.ppc

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
f = Forecaster(horizons=(1, 24), season=24)
pred = f.fit_predict(df, target="temperature", time="time", exog=["pressure", "wind"])
print(f.report())          # MAE / RMSE and skill vs persistence, for every model and horizon
f.save("temp.ppc")

f = Forecaster.load("temp.ppc")
new_pred = f.update(new_rows)   # continues exactly where it stopped
```

Output columns: `<model>_h<H>` is that model's forecast, made on that row, for the value H rows later.
The models are `brain`, `linear`, `persistence`, `seasonal`, `base` (if given) and `auto`.

## How it is scored

Everything is **prequential**: the forecast for row t+H is made at row t, and the models learn from it
only when row t+H arrives. There is no train/test split to leak, so the score table is out-of-sample.
Skill = 1 − MAE / MAE(persistence), where persistence means "no change".

## Benchmarks (honest)

Average absolute error in the second half of each dataset (lower is better):

| data | ahead | brain | linear | no change |
|---|---|---|---|---|
| London temperature (°C), 2016–2025 | 1 h | **0.37** | 0.41 | 0.65 |
| | 24 h | 2.06 | **1.90** | 2.10 |
| Bike-share rentals per hour (UCI) | 1 h | **41** | 59 | 80 |
| | 24 h | 82 | **76** | 78 |
| Household power, kW (UCI) | 1 h | **0.395** | 0.399 | 0.414 |
| London wind 24 h, correcting a professional weather forecast | 24 h | 1.90 | **1.84** | 4.63 (professional forecast alone: 1.99) |

- **The brain shines at short horizons** where inputs interact (bike demand by hour and working day).
- **At 24 h+ a linear model usually wins.** `auto` picks per horizon for you.
- **From history alone it will not beat a physics-based weather model**, but given that forecast with
  `--base` it learns its local biases online.

Reproduce with `python -m experiments.forecast_bench` from the source repository.

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
