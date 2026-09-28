"""Sensor platform: forecast sensors for any numeric sensor with long-term statistics.

configuration.yaml:

    sensor:
      - platform: ppc_forecaster
        source: sensor.house_power        # any sensor with a state_class (so HA keeps hourly statistics)
        name: House power
        mode: mean                        # mean (power, temperature...) or change (energy meters: usage per hour)
        horizons: [1, 6, 24]              # hours ahead, one forecast sensor each
        season: [24, 168]                 # daily + weekly cycles
        history_days: 365

Creates sensor.<name>_forecast_<H>h with the forecast as state and attributes lower / upper (80% interval),
for_time, skill_vs_no_change and interval_coverage (how often the truth fell inside the interval so far).
The model trains in the background at startup, learns from every new hour at :15 past, and is saved in
<config>/.storage so a restart continues where it stopped.
"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import timedelta

import pandas as pd
import voluptuous as vol

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.components.sensor import PLATFORM_SCHEMA, SensorEntity, SensorStateClass
from homeassistant.const import CONF_NAME, EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_time_change
from homeassistant.util import dt as dt_util, slugify

from .core import Runner, stats_to_frame

_LOGGER = logging.getLogger(__name__)

CONF_SOURCE, CONF_MODE, CONF_HORIZONS = "source", "mode", "horizons"
CONF_SEASON, CONF_HISTORY_DAYS, CONF_FAST = "season", "history_days", "fast"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required(CONF_SOURCE): cv.entity_id,
    vol.Optional(CONF_NAME): cv.string,
    vol.Optional(CONF_MODE, default="mean"): vol.In(["mean", "change"]),
    vol.Optional(CONF_HORIZONS, default=[1, 24]): vol.All(cv.ensure_list, [vol.All(vol.Coerce(int), vol.Range(1, 168))]),
    vol.Optional(CONF_SEASON, default=[24, 168]): vol.All(cv.ensure_list, [vol.All(vol.Coerce(int), vol.Range(2, 8760))]),
    vol.Optional(CONF_HISTORY_DAYS, default=365): vol.All(vol.Coerce(int), vol.Range(7, 3650)),
    vol.Optional(CONF_FAST, default=True): cv.boolean,
})


async def async_setup_platform(hass: HomeAssistant, config, async_add_entities, discovery_info=None):
    source = config[CONF_SOURCE]
    name = config.get(CONF_NAME) or source.split(".", 1)[1].replace("_", " ")
    hub = ForecastHub(hass, source, name, config[CONF_MODE], config[CONF_HORIZONS], config[CONF_SEASON],
                      config[CONF_HISTORY_DAYS], config[CONF_FAST])
    hass.data.setdefault("ppc_forecaster", {})[source] = hub
    async_add_entities(hub.entities)

    async def _start(_event=None):
        await hub.async_start()

    if hass.state == CoreState.running:
        hass.async_create_task(_start())
    else:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _start)


class ForecastHub:
    """Loads statistics, runs the Runner in the executor, pushes results to the sensors."""

    def __init__(self, hass, source, name, mode, horizons, seasons, history_days, fast):
        self.hass, self.source, self.name, self.mode = hass, source, name, mode
        self.history_days = history_days
        self.runner = Runner(horizons, seasons, fast)
        self.entities = [ForecastSensor(self, h) for h in horizons]
        self.path = hass.config.path(".storage", f"ppc_forecaster.{slugify(source)}.pkl")
        self.unit = None

    async def _stats(self, start):
        types = {"change"} if self.mode == "change" else {"mean"}
        if self.mode == "change":
            types |= {"sum"}
        rows = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period, self.hass, start, None, {self.source}, "hour", None, types)
        df = stats_to_frame(rows.get(self.source, []), self.mode)
        # HA can return a row for the hour still in progress (built from partial data): only learn from
        # hours that have finished
        return df[df.time + pd.Timedelta(hours=1) <= pd.Timestamp(dt_util.utcnow())].reset_index(drop=True)

    async def async_start(self):
        state = self.hass.states.get(self.source)
        self.unit = state.attributes.get("unit_of_measurement") if state else None
        if self.mode == "change" and self.unit and "/h" not in self.unit:
            self.unit = f"{self.unit}/h"
        loaded = await self.hass.async_add_executor_job(self._load)
        if loaded:
            self.runner = loaded
            df = await self._stats(dt_util.utcnow() - timedelta(days=self.history_days))
            await self.hass.async_add_executor_job(self.runner.update, df)
        else:
            df = await self._stats(dt_util.utcnow() - timedelta(days=self.history_days))
            try:
                await self.hass.async_add_executor_job(self.runner.fit, df)
            except ValueError as err:
                _LOGGER.warning("%s: %s; will retry every hour", self.source, err)
        await self._publish()
        async_track_time_change(self.hass, self._hourly, minute=15, second=0)

    async def _hourly(self, _now):
        if self.runner.model is None:                        # not enough history at startup: try again
            df = await self._stats(dt_util.utcnow() - timedelta(days=self.history_days))
            try:
                await self.hass.async_add_executor_job(self.runner.fit, df)
            except ValueError:
                return
        else:
            df = await self._stats(self.runner.last_time.to_pydatetime() - timedelta(hours=2))
            P = await self.hass.async_add_executor_job(self.runner.update, df)
            _LOGGER.debug("%s: %d statistics rows, %s new hours, model now at %s", self.source, len(df),
                          0 if P is None else len(P), self.runner.last_time)
        await self._publish()

    async def _publish(self):
        if self.runner.model is None:
            return
        latest = await self.hass.async_add_executor_job(self.runner.latest)
        await self.hass.async_add_executor_job(self._save)
        for e in self.entities:
            e.set_forecast(latest.get(e.horizon))

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)   # .storage may not exist yet on a fresh install
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self.runner, fh)
        os.replace(tmp, self.path)

    def _load(self):
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "rb") as fh:
                r = pickle.load(fh)
            if tuple(r.horizons) != tuple(self.runner.horizons) or list(r.seasons) != list(self.runner.seasons):
                return None                                  # settings changed: retrain
            return r
        except Exception as err:                             # corrupt/old file: retrain rather than fail
            _LOGGER.warning("could not load %s (%s); retraining", self.path, err)
            return None


class ForecastSensor(SensorEntity):
    _attr_should_poll = False
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:crystal-ball"

    def __init__(self, hub, horizon):
        self.hub, self.horizon = hub, horizon
        self._attr_name = f"{hub.name} forecast {horizon}h"
        self._attr_unique_id = f"ppc_forecaster_{slugify(hub.source)}_{horizon}h"
        self._attr_available = False
        self._attr_extra_state_attributes = {"source": hub.source}

    @property
    def native_unit_of_measurement(self):
        return self.hub.unit

    @callback
    def set_forecast(self, f):
        if not f or f["value"] is None:
            return
        self._attr_native_value = round(f["value"], 3)
        self._attr_available = True
        self._attr_extra_state_attributes = {"source": self.hub.source, "hours_ahead": self.horizon, **{
            k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if k != "value"}}
        if self.hass is not None:
            self.async_write_ha_state()
