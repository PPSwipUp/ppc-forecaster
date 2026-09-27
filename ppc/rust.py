"""RustBrain: the PPC brain running in Rust (ppc_rs/), behind the same interface as ppc.Brain.

    rb = RustBrain.from_numpy(brain)        # lossless conversion of a NumPy brain (new or trained)
    preds = rb.step(x, signals)             # one tick, same dict in/out as Brain.step
    preds = rb.run(X, signals)              # a block of ticks: X [T, D], signals {head: [T, dim]}
    rb.sleep(0.5); rb.save(path); pickle works

Brains are still initialised by the NumPy code; Rust takes over from there.  Build the
extension with `cargo build --release` in ppc_rs/ and copy target/release/libppc_rs.dylib to
ppc/ppc_rs.abi3.so (scripts: v2/brain/ppc_rs/build.sh).
"""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np

from . import ppc_rs
from .heads import GVFHead


def _l(a):
    return None if a is None else np.asarray(a, float).ravel().tolist()


def _scale(s):
    return {"mu": _l(s.mu), "var": _l(s.var), "n": int(s.n), "tau": float(s.tau), "center": bool(s.center)}


def _head(h):
    if isinstance(h, GVFHead):
        return {"Gvf": {"name": h.name, "gamma": float(h.gamma), "lam": float(h.lam), "dim": int(h.dim),
                        "lr": float(h.lr), "core_weight": float(h.core_weight), "w": _l(h.w),
                        "trace": _l(h.trace), "phi_prev": _l(h.phi_prev), "scale": _scale(h.scale),
                        "pp": float(h.pp)}}
    rls = getattr(h, "solver", "nlms") == "rls"
    return {"Predict": {"name": h.name, "dim": int(h.dim), "delay": int(h.delay), "binary": h.kind == "binary",
                        "lr": float(h.lr), "core_weight": float(h.core_weight), "rls": rls,
                        "lam": float(getattr(h, "lam", 1.0)), "p": _l(getattr(h, "P", None)), "w": _l(h.w),
                        "buf": [_l(b) for b in h.buf], "scale": _scale(h.scale)}}


def to_json(brain):
    c = brain.core
    defer = bool(getattr(c, "defer", False))
    dw, da = (c._deferred() if defer else (None, None))
    m = brain.mod
    state = {
        "core": {"n": int(c.N), "d": int(c.D), "m": int(c.M), "alpha": _l(c.alpha), "w": _l(c.W), "f": _l(c.F),
                 "a": _l(c.A), "e": _l(c.E), "e_prev": _l(c.E_prev), "gw": float(c.gW), "ga": float(c.gA),
                 "h": _l(c.h), "h_mean": _l(c.h_mean), "z": _l(c.z), "fast_decay": float(c.fast_decay),
                 "hebb_lr": float(c.hebb_lr), "slow_lr": float(c.slow_lr), "plast_lr": float(c.plast_lr),
                 "init_plasticity": float(c.init_plasticity), "input_scale": float(c.input_scale),
                 "track": bool(getattr(c, "track", True)), "defer": defer,
                 "dw": _l(dw) if defer else [], "da": _l(da) if defer else []},
        "heads": [_head(h) for h in brain.heads.values()],
        "modulator": {"rf": float(m.rf), "rs": float(m.rs), "k": float(m.k), "lo": float(m.lo), "hi": float(m.hi),
                      "fast": None if m.fast is None else float(m.fast),
                      "slow": None if m.slow is None else float(m.slow), "n": int(m.n), "gain": float(m.gain)},
        "inp": _scale(brain.inp), "input_clip": float(brain.input_clip),
        "plastic": bool(brain.plastic), "hebbian": bool(brain.hebbian), "modulation": bool(brain.modulation),
        "neurogenesis": bool(brain.neurogenesis), "sleep_only": bool(getattr(brain, "sleep_only", False)),
        "replace_rate": float(brain.replace_rate), "maturity": int(brain.maturity),
        "utility": _l(brain.utility), "age": [int(a) for a in brain.age],
        "birth_debt": float(brain._birth_debt), "t": int(brain.t), "births": int(brain.births),
    }
    return json.dumps(state, allow_nan=False)


class RustBrain:
    def __init__(self, inner):
        self._b = inner
        self._names = inner.head_names()
        self.heads = dict(zip(self._names, inner.head_dims()))

    @classmethod
    def from_numpy(cls, brain, seed=0):
        return cls(ppc_rs.Brain.from_json(to_json(brain), seed))

    N = property(lambda self: self._b.n)
    births = property(lambda self: self._b.births)
    t = property(lambda self: self._b.t)
    mod = property(lambda self: SimpleNamespace(gain=self._b.gain))

    def _sig_list(self, signals):
        signals = signals or {}
        out = []
        for name in self._names:
            v = signals.get(name)
            out.append(None if v is None else np.ascontiguousarray(v, dtype=float))
        return out

    def step(self, x, signals=None):
        p = self._b.step(np.ascontiguousarray(x, dtype=float), self._sig_list(signals))
        return dict(zip(self._names, p))

    def run(self, X, signals=None):
        p = self._b.run(np.ascontiguousarray(X, dtype=float), self._sig_list(signals))
        return dict(zip(self._names, p))

    def sleep(self, frac=0.5):
        self._b.sleep(frac)

    def stats(self):
        t, births, gain, w, f, a = self._b.stats()
        return {"t": t, "births": births, "gain": gain, "|W|": w, "|F|": f, "A": a}

    def get(self, what):
        return np.asarray(self._b.get(what))

    def __getstate__(self):
        return {"bytes": self._b.to_bytes()}

    def __setstate__(self, st):
        self.__init__(ppc_rs.Brain.from_bytes(st["bytes"]))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
