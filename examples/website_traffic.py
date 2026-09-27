"""Forecast daily Wikipedia page views (a stand-in for website traffic) 1 and 7 days ahead.

Page views are spiky (news events), and this example shows why `auto` exists: at 1 day the learners beat
"no change", but at 7 days the spikes throw them off and `auto` falls back to the naive forecast.

  python examples/website_traffic.py ["Article_title"]
"""
import sys

from _fetch import wikipedia_views
from ppc.forecaster import Forecaster

article = sys.argv[1] if len(sys.argv) > 1 else "Python_(programming_language)"
df = wikipedia_views(article)
f = Forecaster(horizons=(1, 7), season=7)
f.fit_predict(df, target="views", time="time")
print(f"{article}: {len(df)} days\n")
print(f.report()[["horizon", "model", "MAE_2nd_half", "skill_2nd_half"]].round({"MAE_2nd_half": 1, "skill_2nd_half": 3}).to_string(index=False))
