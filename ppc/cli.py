"""ppc -- command-line forecaster that keeps learning.

  ppc train  data.csv --target temp [--time time] [--horizons 1,24] [--season 24,168] [--exog all|a,b]
                      [--base 24:nwp_col] [--transform auto|log|none] [--interval 0.8] [--fast]
                      [--save model.ppc] [--out forecasts.csv]
      Streams the whole file through every model (each forecast made before its answer is seen),
      prints an honest score table against persistence / seasonal-naive / linear baselines,
      and optionally saves the model and the forecasts.  --base gives it an existing forecast to improve.
  ppc update model.ppc new.csv [--out forecasts.csv]
      Continues learning on new rows and prints/saves their forecasts.
  ppc report model.ppc
      Re-prints the score table of everything the model has seen.

Run as `python -m ppc.cli ...` or, once installed, `ppc ...`.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from .forecaster import Forecaster
from .panel import PanelForecaster


def _table(f):
    R = f.report()
    pd.set_option("display.width", 200)
    cols = ["horizon", "model", "MAE_all", "RMSE_all", "skill_all", "MAE_2nd_half", "skill_2nd_half"]
    cols += [c for c in ("coverage_2nd_half", "width_2nd_half", "series", "wins") if c in R]
    print(R[cols].round(4).to_string(index=False, na_rep=""))
    print("\nskill = 1 - MAE / MAE(persistence): >0 beats 'no change', higher is better.")
    if "wins" in R:
        print("panel: MAE/skill are averages over series; wins = series where that model was most accurate.")
    if "coverage_2nd_half" in R:
        print(f"coverage = share of values inside auto's {f.interval:.0%} prediction interval (should be ~{f.interval:.0%}).")
    for h in f.horizons:
        r = R[R.horizon == h].set_index("model")
        best = r.drop(index="auto", errors="ignore").skill_2nd_half.idxmax()
        cmp = ""
        if "brain" in r.index:
            b, l = r.loc["brain", "MAE_2nd_half"], r.loc["linear", "MAE_2nd_half"]
            cmp = f";  brain vs linear MAE {100 * (b / l - 1):+.1f}%"
        print(f"h={h}: best in 2nd half = {best}{cmp}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ppc", description="Continual-learning forecaster (PPC brain)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("csv")
    t.add_argument("--target", required=True)
    t.add_argument("--time")
    t.add_argument("--horizons", default="1")
    t.add_argument("--season", default="", help="seasonal cycle length(s) in rows, e.g. 24 or 24,168")
    t.add_argument("--exog", default="", help="'all' for every other numeric column, or a,b,c")
    t.add_argument("--neurons", type=int, default=128)
    t.add_argument("--transform", default="auto", choices=["auto", "log", "none"],
                   help="auto: learn on log(1+y) for skewed non-negative targets (default)")
    t.add_argument("--interval", type=float, default=0.8, help="prediction-interval level, 0 to turn off")
    t.add_argument("--fast", action="store_true", help="brain only for horizons <= 6 (much faster, usually as accurate)")
    t.add_argument("--id", help="column naming the series, to forecast many series (stores, sensors...) at once")
    t.add_argument("--base", default="", help="existing forecast to improve, as H:column (row t = its forecast for t+H)")
    t.add_argument("--save")
    t.add_argument("--out")
    u = sub.add_parser("update")
    u.add_argument("model")
    u.add_argument("csv")
    u.add_argument("--out")
    r = sub.add_parser("report")
    r.add_argument("model")
    a = ap.parse_args(argv)

    if a.cmd == "train":
        df = pd.read_csv(a.csv)
        if a.target not in df:
            sys.exit(f"error: target column '{a.target}' not in {list(df.columns)}")
        if a.time and a.time not in df:
            sys.exit(f"error: time column '{a.time}' not in {list(df.columns)}")
        if a.time:
            df = df.sort_values(a.time, kind="stable")
        if a.id and a.id not in df:
            sys.exit(f"error: id column '{a.id}' not in {list(df.columns)}")
        if a.exog == "all":
            exog = [c for c in df.select_dtypes("number").columns if c not in (a.target, a.time, a.id)]
        else:
            exog = [c for c in a.exog.split(",") if c]
            missing = [c for c in exog if c not in df]
            if missing:
                sys.exit(f"error: exog columns not found: {missing}")
        hs = [int(h) for h in a.horizons.split(",")]
        base = {}
        for item in filter(None, a.base.split(",")):
            h, _, col = item.partition(":")
            if col not in df or int(h) not in hs:
                sys.exit(f"error: --base {item}: need H in --horizons and an existing column")
            base[int(h)] = col
        try:
            seasons = [int(x) for x in a.season.split(",") if x]
        except ValueError:
            sys.exit(f"error: --season must be whole numbers like 24 or 24,168, got '{a.season}'")
        if not 0 <= a.interval < 1:
            sys.exit("error: --interval must be between 0 and 1 (e.g. 0.8), or 0 to turn off")
        kw = dict(horizons=hs, season=seasons or None, n_neurons=a.neurons, base=base,
                  transform=a.transform, interval=a.interval or None, fast=a.fast)
        f = PanelForecaster(a.id, **kw) if a.id else Forecaster(**kw)
        t0 = time.time()
        P = f.fit_predict(df, target=a.target, time=a.time, exog=exog)
        if a.id:
            n_log = sum(m.use_log for m in f.models.values())
            print(f"{len(df)} rows, {len(f.models)} series ({n_log} on log scale), {len(exog)} exogenous columns, "
                  f"horizons {hs}, {time.time() - t0:.1f}s\n")
        else:
            print(f"{len(df)} rows, {len(exog)} exogenous columns, horizons {hs}, "
                  f"{'log' if f.use_log else 'raw'} scale, {time.time() - t0:.1f}s\n")
        _table(f)
        if a.out:
            P.to_csv(a.out, index=False)
        if a.save:
            f.save(a.save)
            print(f"saved {a.save}")
    elif a.cmd == "update":
        f = Forecaster.load(a.model)
        df = pd.read_csv(a.csv)
        if f.time:
            df = df.sort_values(f.time, kind="stable")
        P = f.update(df)
        idc = [f.id_col] if isinstance(f, PanelForecaster) else []
        cols = idc + (["time"] if "time" in P else []) + ["y"] + [c for h in f.horizons for c in
                                                             (f"auto_h{h}", f"auto_h{h}_lo", f"auto_h{h}_hi") if c in P]
        print(P[cols].tail(10).to_string(index=False))
        if a.out:
            P.to_csv(a.out, index=False)
        f.save(a.model)
    else:
        _table(Forecaster.load(a.model))


if __name__ == "__main__":
    main()
