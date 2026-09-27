# Benchmark: ppc-forecaster vs River vs statsforecast

`python benchmarks/compare.py` · raw numbers in [`results.csv`](results.csv) · run 2026-09-28 on an Apple M-series Mac,
ppc-forecaster 0.1.0 from PyPI, River 0.26.1, statsforecast 2.1.1.

**Setup.** Three public hourly series, univariate (no extra inputs for anyone):
London temperature 2016–2025 (87,672 h, ERA5 via Open-Meteo), UCI household power 2006–2010 (34,589 h),
UCI bike-share rentals 2011–2012 (17,544 h).

- **One step ahead.** Forecast the next hour, then learn it, every hour. Scored on the second half.
- **Day ahead.** At midnight, forecast the next 24 hours, on each of the last 100 days. Score = mean absolute error over all 2,400 values.
  statsforecast re-fits every day on the previous 56 days.

River's settings were tuned on the first half of each dataset (the half nobody is scored on). ppc used its defaults
(`Forecaster(horizons=..., season=24)`). Lower is better; **bold** = best in its column.

## One hour ahead (mean absolute error, second half)

| method | temperature °C | household kW | bike rentals |
|---|---|---|---|
| ppc brain | **0.385** | 0.397 | **40.0** |
| ppc linear | 0.411 | 0.399 | 58.6 |
| ppc auto | **0.385** | 0.395 | **40.0** |
| River SNARIMAX | 0.416 | 0.417 | 52.1 |
| River Holt-Winters | 0.611 | 0.419 | 65.2 |
| River linear regression on lags | 0.450 | **0.392** | 46.5 |
| naive (last value) | 0.653 | 0.414 | 79.8 |
| seasonal naive (same hour yesterday) | 2.094 | 0.585 | 76.8 |

## Next 24 hours from midnight (mean absolute error, last 100 days)

| method | temperature °C | household kW | bike rentals |
|---|---|---|---|
| ppc brain | 1.482 | 0.537 | 87.8 |
| ppc linear | 1.455 | **0.503** | 95.0 |
| ppc auto | **1.437** | 0.505 | 77.6 |
| River SNARIMAX | 1.880 | 0.746 | 103.8 |
| River Holt-Winters | 1.637 | 0.645 | 98.1 |
| River linear regression on lags | 1.723 | 0.516 | 73.1 |
| statsforecast MSTL (daily + weekly) | 1.537 | 0.530 | **60.8** |
| statsforecast AutoETS | 1.761 | 0.703 | 83.1 |
| seasonal naive | 2.133 | 0.537 | 77.0 |

## What this says (honestly)

- **One hour ahead, ppc is best or tied on all three.** The brain's edge is largest where inputs interact:
  bike rentals, 14% below the best River model and 50% below naive.
- **Day ahead, ppc is best on temperature and household power**, beating MSTL, which re-fits every day.
- **On bike rentals day-ahead, statsforecast MSTL wins clearly (60.8 vs 77.6).** Rentals have a strong
  *weekly* shape, which MSTL models explicitly (season 24 and 168) and ppc here was only told about the daily one.
- **ppc is slower:** 13–67 s per dataset for 24 horizons × 2 models, against about 1 s for River. It is still far more
  than fast enough for live hourly data.
- AutoARIMA was left out: 20–60 s per fit, times 300 daily re-fits, was over an hour.
- **Not tested here:** extra inputs (weather columns, calendars) and the `base` option, which are ppc's other strengths
  (see the main README).
