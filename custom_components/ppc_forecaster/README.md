# PPC Forecaster for Home Assistant

Forecast sensors for anything Home Assistant keeps long-term statistics for: house power, energy use,
temperature, solar output. Each forecast comes with an 80% range and a score showing how it has done
on your own data so far.

- Trains on your hourly statistics history at startup (HA keeps these forever for sensors with a `state_class`).
- Learns from every new hour at :15 past the hour. No retraining, no cloud.
- Saves the model in `.storage`, so a restart continues where it stopped.

## Install

**HACS:** add `https://github.com/PPSwipUp/ppc-forecaster` as a custom repository (type: Integration),
install *PPC Forecaster*, restart.

**Manual:** copy `custom_components/ppc_forecaster` into your `config/custom_components/` folder and restart.

Home Assistant installs the `ppc-forecaster` Python package itself. Prebuilt wheels cover x86_64 and
aarch64 Linux, glibc and musl (Home Assistant OS on a NUC or 64-bit Raspberry Pi, or Docker).

## Configure

In `configuration.yaml`:

```yaml
sensor:
  - platform: ppc_forecaster
    source: sensor.house_power      # any sensor with a state_class
    name: House power
    mode: mean                      # mean: power, temperature...  change: meters that count up (kWh) -> use per hour
    horizons: [1, 6, 24]            # hours ahead; one sensor each
    season: [24, 168]               # daily + weekly cycles (the default)
    history_days: 365               # how much history to train on (default 365)
    fast: true                      # default; false also runs the network for horizons over 6 h
```

This creates `sensor.house_power_forecast_1h`, `_6h` and `_24h`.

| attribute | meaning |
|---|---|
| state | the forecast |
| `lower`, `upper` | 80% range: the real value should land inside about 8 times in 10 |
| `for_time` | the hour being forecast (UTC) |
| `skill_vs_no_change` | 0.3 means 30% smaller errors than assuming "same as last hour"; ≤ 0 means no better |
| `interval_coverage` | how often the real value has actually landed inside the range |

It needs at least three weeks of hourly history (three times the longest season). With less, it waits and
retries every hour.

## Ideas

- Run the dishwasher or charge the car when the 6 h power forecast is low.
- Pre-heat when tomorrow morning's temperature forecast is cold.
- Alert when real use leaves the forecast range (something left on).

## Limits

- Hourly only; forecasts are refreshed once an hour.
- Learns from the sensor's own history. It doesn't see weather forecasts, so it can't anticipate a
  sudden cold snap. It learns the daily and weekly pattern and the recent level.
