# Benchmark: ppc-forecaster vs River vs statsforecast

`python benchmarks/compare.py` · raw numbers in [`results.csv`](results.csv) · run 2026-09-28 on an Apple M-series Mac,
ppc-forecaster 0.2.0, River 0.26.1, statsforecast 2.1.1.

**Setup.** Three public hourly series, univariate (no extra inputs for anyone):
London temperature 2016–2025 (87,672 h, ERA5 via Open-Meteo), UCI household power 2006–2010 (34,589 h),
UCI bike-share rentals 2011–2012 (17,544 h).

- **One step ahead.** Forecast the next hour, then learn it, every hour. Scored on the second half.
- **Day ahead.** At midnight, forecast the next 24 hours, on each of the last 100 days. Score = mean absolute error over all 2,400 values.
  statsforecast re-fits every day on the previous 56 days.

River's settings were tuned on the first half of each dataset (the half nobody is scored on). ppc used its defaults
with daily and weekly seasons (`Forecaster(horizons=..., season=[24, 168])`). Lower is better; **bold** = best in its column.

## One hour ahead (mean absolute error, second half)

| method | temperature °C | household kW | bike rentals |
|---|---|---|---|
| ppc brain | **0.388** | **0.361** | **32.2** |
| ppc linear | 0.393 | 0.365 | 34.0 |
| ppc auto | **0.388** | **0.361** | 32.3 |
| River SNARIMAX | 0.416 | 0.417 | 52.1 |
| River Holt-Winters | 0.611 | 0.419 | 65.2 |
| River linear regression on lags | 0.450 | 0.392 | 46.5 |
| naive (last value) | 0.653 | 0.414 | 79.8 |
| seasonal naive (same hour yesterday) | 2.094 | 0.585 | 76.8 |

## Next 24 hours from midnight (mean absolute error, last 100 days)

| method | temperature °C | household kW | bike rentals |
|---|---|---|---|
| ppc brain | 1.460 | 0.478 | 63.9 |
| ppc linear | 1.368 | 0.476 | **60.6** |
| ppc auto | **1.358** | **0.469** | 62.6 |
| River SNARIMAX | 1.880 | 0.746 | 103.8 |
| River Holt-Winters | 1.637 | 0.645 | 98.1 |
| River linear regression on lags | 1.723 | 0.516 | 73.1 |
| statsforecast MSTL (daily + weekly) | 1.537 | 0.530 | 60.8 |
| statsforecast AutoETS | 1.761 | 0.703 | 83.1 |
| seasonal naive | 2.133 | 0.537 | 77.0 |

**Prediction intervals.** ppc's 80% intervals held 79–80% of the actual values on all three datasets, at both
1 h and 24 h.

## What this says (honestly)

- **One hour ahead, ppc is best on all three**, by 7–31% over the best River model (bike rentals 32 vs 46.5).
- **Day ahead, ppc beats everything on temperature (−12% vs MSTL) and household power (−11%)**, and ties
  statsforecast MSTL on bike rentals (linear 60.6, auto 62.6, MSTL 60.8), even though MSTL re-fits every day.
- **What changed from 0.1** (where MSTL won bike rentals 60.8 vs 77.6): version 0.2 accepts several seasons
  (`season=[24, 168]`), adds "same hour last week" features, and learns on a log scale when the data is skewed
  counts. Household power and bike rentals switched to the log scale automatically; temperature (which can go
  negative) did not.
- **ppc is slower:** 15–74 s per dataset for 24 horizons × 2 models, against about 1 s for River. It is still far more
  than fast enough for live hourly data.
- AutoARIMA was left out: 20–60 s per fit, times 300 daily re-fits, was over an hour.
- **Not tested here:** extra inputs (weather columns, calendars) and the `base` option, which are ppc's other strengths
  (see the main README).
