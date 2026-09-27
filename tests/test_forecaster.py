"""Package tests for ppc-forecaster (run by CI against each built wheel).  Self-contained: synthetic data only."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ppc.forecaster import Forecaster


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    x = rng.standard_normal(n).cumsum() * 0.1
    y = 10 * np.sin(2 * np.pi * np.arange(n) / 24) + x + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"time": t, "y": y, "x": x})


def test_learners_beat_persistence():
    f = Forecaster(horizons=(1, 6), season=24)
    f.fit_predict(synthetic(), target="y", time="time", exog=["x"])
    R = f.report().set_index(["horizon", "model"])
    for h in (1, 6):
        for m in ("brain", "linear", "auto"):
            assert R.loc[(h, m), "skill_2nd_half"] > 0.5, (h, m)


def test_forecasts_do_not_peek():
    """changing a future value must not change any forecast made before it"""
    df = synthetic(800)
    a = Forecaster(horizons=(1,), season=24).fit_predict(df, target="y", time="time")
    df2 = df.copy()
    df2.loc[700, "y"] += 100.0
    b = Forecaster(horizons=(1,), season=24).fit_predict(df2, target="y", time="time")
    for c in ("brain_h1", "linear_h1", "auto_h1"):
        np.testing.assert_allclose(a[c].values[:700], b[c].values[:700])


def test_save_and_resume_is_exact(tmp_path):
    df = synthetic(2000)
    full = Forecaster(horizons=(1, 6), season=24).fit_predict(df, target="y", time="time", exog=["x"])
    f = Forecaster(horizons=(1, 6), season=24)
    f.fit_predict(df.iloc[:1500], target="y", time="time", exog=["x"])
    f.save(tmp_path / "m.ppc")
    part = Forecaster.load(tmp_path / "m.ppc").update(df.iloc[1500:])
    for c in ("brain_h1", "linear_h6", "auto_h1", "auto_h6"):
        np.testing.assert_allclose(part[c].values, full[c].values[1500:], atol=1e-9)


def test_base_forecast_is_used():
    """given a near-perfect base forecast, the corrected models should be about as good as it"""
    df = synthetic(3000)
    df["good"] = df.y.shift(-6) + 0.05 * np.random.default_rng(1).standard_normal(len(df))
    f = Forecaster(horizons=(6,), season=24, base={6: "good"})
    f.fit_predict(df, target="y", time="time")
    R = f.report().set_index("model")
    assert R.loc["base", "MAE_2nd_half"] < 0.1
    assert R.loc["linear", "MAE_2nd_half"] < 0.2


def test_missing_input_is_flagged_not_zeroed():
    df = synthetic(2000)
    df.loc[:999, "x"] = np.nan                      # input starts half-way
    f = Forecaster(horizons=(1,), season=24)
    f.fit_predict(df, target="y", time="time", exog=["x"])
    assert f.miss == {"x": True}
    assert np.isfinite(f.log.brain_h1).all()


def test_cli_train_update_report(tmp_path):
    df = synthetic(1500)
    df.iloc[:1000].to_csv(tmp_path / "a.csv", index=False)
    df.iloc[1000:].to_csv(tmp_path / "b.csv", index=False)
    run = lambda *a: subprocess.run([sys.executable, "-m", "ppc.cli", *a], capture_output=True, text=True)
    r = run("train", str(tmp_path / "a.csv"), "--target", "y", "--time", "time", "--horizons", "1",
            "--season", "24", "--exog", "all", "--save", str(tmp_path / "m.ppc"))
    assert r.returncode == 0, r.stderr
    assert "persistence" in r.stdout
    r = run("update", str(tmp_path / "m.ppc"), str(tmp_path / "b.csv"), "--out", str(tmp_path / "p.csv"))
    assert r.returncode == 0, r.stderr
    assert len(pd.read_csv(tmp_path / "p.csv")) == 500
    r = run("report", str(tmp_path / "m.ppc"))
    assert r.returncode == 0 and "auto" in r.stdout
    r = run("train", str(tmp_path / "a.csv"), "--target", "nope")
    assert r.returncode != 0 and "not in" in r.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
