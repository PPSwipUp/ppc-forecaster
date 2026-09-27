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
    for c in ("brain_h1", "linear_h6", "auto_h1", "auto_h6", "auto_h1_lo", "auto_h6_hi"):
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


def weekly(n=24 * 7 * 40, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    day = np.sin(2 * np.pi * i / 24)
    week = np.where((i // 24) % 7 >= 5, -1.0, 0.5)             # weekends are different
    y = 5 * day + 4 * week + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="h"), "y": y})


def test_intervals_cover_about_right():
    f = Forecaster(horizons=(1, 6), season=24, interval=0.8)
    f.fit_predict(synthetic(4000), target="y", time="time")
    R = f.report().set_index(["horizon", "model"])
    for h in (1, 6):
        assert 0.72 < R.loc[(h, "auto"), "coverage_2nd_half"] < 0.88, R.loc[(h, "auto")]


def test_weekly_season_helps_day_ahead():
    df = weekly()
    one = Forecaster(horizons=(24,), season=24)
    one.fit_predict(df, target="y", time="time")
    two = Forecaster(horizons=(24,), season=[24, 168])
    two.fit_predict(df, target="y", time="time")
    m1 = one.report().set_index("model").loc["auto", "MAE_2nd_half"]
    R2 = two.report().set_index("model")
    assert "seasonal168" in R2.index
    assert R2.loc["auto", "MAE_2nd_half"] < 0.9 * m1


def test_log_transform_for_skewed_counts():
    rng = np.random.default_rng(0)
    n = 4000
    lam = np.exp(2 + 2.5 * np.sin(2 * np.pi * np.arange(n) / 24))     # skewed counts
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="h"), "y": rng.poisson(lam)})
    f = Forecaster(horizons=(1,), season=24)                     # transform="auto"
    P = f.fit_predict(df, target="y", time="time")
    assert f.use_log
    assert np.isfinite(P.brain_h1).all() and (P.brain_h1 > -1).all()
    g = Forecaster(horizons=(1,), season=24, transform=None)
    g.fit_predict(df, target="y", time="time")
    assert not g.use_log
    mae = lambda m: m.report().set_index("model").loc["auto", "MAE_2nd_half"]
    assert mae(f) < mae(g)                                       # learning on the log scale helps here


def test_models_saved_by_0_1_still_load(tmp_path):
    import pickle
    df = synthetic(1200)
    f = Forecaster(horizons=(1,), season=24)
    f.fit_predict(df.iloc[:1000], target="y", time="time")
    st = dict(f.__dict__)
    for k in ("seasons", "transform", "use_log", "interval", "interval_window", "resid", "pend_int"):
        st.pop(k)                                                # what a 0.1.x pickle looked like
    old = Forecaster.__new__(Forecaster)
    old.__dict__.update(st)
    (tmp_path / "old.ppc").write_bytes(pickle.dumps(old))
    P = Forecaster.load(tmp_path / "old.ppc").update(df.iloc[1000:])
    assert len(P) == 200 and np.isfinite(P.auto_h1).all()


def test_cli_train_update_report(tmp_path):
    df = synthetic(1500)
    df.iloc[:1000].to_csv(tmp_path / "a.csv", index=False)
    df.iloc[1000:].to_csv(tmp_path / "b.csv", index=False)
    # run from tmp_path so `-m ppc.cli` imports the installed package, not a source checkout in the cwd
    run = lambda *a: subprocess.run([sys.executable, "-m", "ppc.cli", *a], capture_output=True, text=True,
                                    cwd=tmp_path)
    r = run("train", str(tmp_path / "a.csv"), "--target", "y", "--time", "time", "--horizons", "1",
            "--season", "24,168", "--exog", "all", "--save", str(tmp_path / "m.ppc"))
    assert r.returncode == 0, r.stderr
    assert "persistence" in r.stdout and "coverage" in r.stdout
    r = run("update", str(tmp_path / "m.ppc"), str(tmp_path / "b.csv"), "--out", str(tmp_path / "p.csv"))
    assert r.returncode == 0, r.stderr
    assert len(pd.read_csv(tmp_path / "p.csv")) == 500
    r = run("report", str(tmp_path / "m.ppc"))
    assert r.returncode == 0 and "auto" in r.stdout
    r = run("train", str(tmp_path / "a.csv"), "--target", "nope")
    assert r.returncode != 0 and "not in" in r.stderr
    r = run("train", str(tmp_path / "a.csv"), "--target", "y", "--season", "day")
    assert r.returncode != 0 and "--season" in r.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
