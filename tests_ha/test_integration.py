"""End-to-end test of the Home Assistant integration in a real (test) Home Assistant with a real recorder.
Run from the repository root:  pytest tests_ha
Needs: pip install pytest-homeassistant-custom-component ppc-forecaster
"""
import os
from datetime import timedelta

import numpy as np
import pytest
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed
from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

from conftest import use_our_custom_components

SRC = "sensor.test_power"


def power(t):
    """a house: daily shape, quieter weekends, noise (W)"""
    h, dow = t.hour, t.weekday()
    base = 400 + 300 * np.sin(2 * np.pi * (h - 7) / 24) + (150 if dow >= 5 else 0)
    return float(base + np.random.default_rng(int(t.timestamp())).normal(0, 40))


def metadata():
    md = {"has_mean": True, "has_sum": False, "name": "Test power", "source": "recorder",
          "statistic_id": SRC, "unit_of_measurement": "W"}
    try:                                                   # HA >= 2025.x adds mean_type / unit_class
        from homeassistant.components.recorder.models import StatisticMeanType
        md["mean_type"] = StatisticMeanType.ARITHMETIC
        md["unit_class"] = "power"
    except ImportError:
        pass
    return md


def import_hours(hass, start, n):
    rows = []
    for i in range(n):
        t = start + timedelta(hours=i)
        v = power(t)
        rows.append({"start": t, "mean": v, "min": v, "max": v})
    async_import_statistics(hass, metadata(), rows)


CONFIG = {"sensor": [{"platform": "ppc_forecaster", "source": SRC, "name": "Test power",
                      "horizons": [1, 24], "season": [24, 168], "history_days": 90}]}


async def setup(hass):
    assert await async_setup_component(hass, "sensor", CONFIG)
    await hass.async_block_till_done()
    for _ in range(3):                                     # training runs in the executor
        await hass.async_block_till_done()
        await async_wait_recording_done(hass)


@pytest.mark.parametrize("expected_lingering_timers", [True])
async def test_forecast_sensors(recorder_mock, enable_custom_integrations, hass, expected_lingering_timers, freezer):
    use_our_custom_components()
    path = hass.config.path(".storage", "ppc_forecaster.sensor_test_power.pkl")
    if os.path.exists(path):                               # the harness's config folder outlives each run
        os.remove(path)
    now = dt_util.utcnow().replace(minute=0, second=0, microsecond=0)
    freezer.move_to(now + timedelta(minutes=20))
    start = now - timedelta(days=60)
    import_hours(hass, start, 60 * 24)
    await async_wait_recording_done(hass)
    hass.states.async_set(SRC, "500", {"unit_of_measurement": "W", "state_class": "measurement"})

    await setup(hass)

    s1 = hass.states.get("sensor.test_power_forecast_1h")
    s24 = hass.states.get("sensor.test_power_forecast_24h")
    assert s1 is not None and s24 is not None, "forecast sensors were not created"
    v1 = float(s1.state)
    a = s1.attributes
    assert 0 < v1 < 1500
    assert a["lower"] < v1 < a["upper"]
    assert a["unit_of_measurement"] == "W"
    assert a["skill_vs_no_change"] > 0.2                   # clearly better than 'no change' on this signal
    assert 0.6 < a["interval_coverage"] < 0.95
    first_for = a["for_time"]
    hub = hass.data["ppc_forecaster"][SRC]
    assert hub.runner.last_time == now - timedelta(hours=1)     # the unfinished current hour is not learned from

    # a new hour arrives -> at :15 the model learns from it and moves its forecast forward one hour
    import_hours(hass, now, 1)
    await async_wait_recording_done(hass)
    freezer.move_to(now + timedelta(hours=1, minutes=15))
    async_fire_time_changed(hass, now + timedelta(hours=1, minutes=15))
    for _ in range(20):                                    # statistics load in the recorder's own executor
        await setup(hass)
        if hass.states.get("sensor.test_power_forecast_1h").attributes["for_time"] != first_for:
            break
    assert hub.runner.last_time == now
    s1b = hass.states.get("sensor.test_power_forecast_1h")
    assert s1b.attributes["for_time"] == (now + timedelta(hours=1)).isoformat()
    assert s1b.attributes["for_time"] != first_for

    assert os.path.exists(path), "model was not saved"
