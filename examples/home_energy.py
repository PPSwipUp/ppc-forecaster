"""Forecast a home's electricity use 1 hour and 1 day ahead (UCI household power, 2006-2010).

  python examples/home_energy.py
"""
from _fetch import household_power
from ppc.forecaster import Forecaster

df = household_power()
f = Forecaster(horizons=(1, 24), season=24)
f.fit_predict(df, target="kw", time="time")
print(f.report()[["horizon", "model", "MAE_2nd_half", "skill_2nd_half"]].round(3).to_string(index=False))
