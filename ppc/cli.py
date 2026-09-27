"""ppc -- command-line forecaster that keeps learning.

  ppc train  data.csv --target temp [--time time] [--horizons 1,24] [--season 24] [--exog all|a,b]
                      [--base 24:nwp_col] [--save model.ppc] [--out forecasts.csv]
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


def _table(f):
    R = f.report()
    pd.set_option("display.width", 200)
    cols = ["horizon", "model", "MAE_all", "RMSE_all", "skill_all", "MAE_2nd_half", "skill_2nd_half"]
    print(R[cols].round(4).to_string(index=False))
    print("\nskill = 1 - MAE / MAE(persistence): >0 beats 'no change', higher is better.")
    for h in f.horizons:
        r = R[R.horizon == h].set_index("model")
        best = r.MAE_2nd_half.idxmin()
        b, l = r.loc["brain", "MAE_2nd_half"], r.loc["linear", "MAE_2nd_half"]
        print(f"h={h}: best in 2nd half = {best};  brain vs linear MAE {100 * (b / l - 1):+.1f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ppc", description="Continual-learning forecaster (PPC brain)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("csv")
    t.add_argument("--target", required=True)
    t.add_argument("--time")
    t.add_argument("--horizons", default="1")
    t.add_argument("--season", type=int)
    t.add_argument("--exog", default="", help="'all' for every other numeric column, or a,b,c")
    t.add_argument("--neurons", type=int, default=128)
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
        if a.exog == "all":
            exog = [c for c in df.select_dtypes("number").columns if c not in (a.target, a.time)]
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
        f = Forecaster(horizons=hs, season=a.season, n_neurons=a.neurons, base=base)
        t0 = time.time()
        P = f.fit_predict(df, target=a.target, time=a.time, exog=exog)
        print(f"{len(df)} rows, {len(exog)} exogenous columns, horizons {hs}, {time.time() - t0:.1f}s\n")
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
        cols = (["time"] if "time" in P else []) + ["y"] + [f"auto_h{h}" for h in f.horizons]
        print(P[cols].tail(10).to_string(index=False))
        if a.out:
            P.to_csv(a.out, index=False)
        f.save(a.model)
    else:
        _table(Forecaster.load(a.model))


if __name__ == "__main__":
    main()
