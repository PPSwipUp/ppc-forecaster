"""Brain = senses (normaliser) + PPC core + heads + neuromodulator + neurogenesis + sleep.

One call per tick:   predictions = brain.step(x_t, signals_t)

Inside a tick (order matters -- it is what makes the brain unable to cheat):
  1. sense    x_t -> normalise -> core state h_t      (only info available at t)
  2. learn    signals_t resolve predictions made at t-1 .. t-delay; errors update
              readouts, and delay-1 errors are broadcast back into the core
  3. modulate sustained surprise raises plasticity, routine noise leaves it at 1
  4. hebbian  fast weights imprint the current co-activity
  5. renew    lowest-utility mature neuron occasionally replaced (continual backprop)
  6. predict  every head outputs its prediction from h_t

Ablation switches (plastic / hebbian / modulation / neurogenesis) and n_neurons=0 turn the
same code into baselines: n_neurons=0 is plain online linear regression; plastic=False,
hebbian=False, neurogenesis=False is a frozen reservoir (echo-state net) with online readout.
"""
from __future__ import annotations

import pickle

import numpy as np

from .core import PPCCore
from .heads import _Scale


class Modulator:
    """Tonic neuromodulator: gain = exp(k * log(fast_err / slow_err)), clipped.
    Detects a *sustained* rise in error (regime change), not single outliers."""

    def __init__(self, fast_tau=500.0, slow_tau=10000.0, k=0.5, lo=0.5, hi=2.0):
        self.rf, self.rs, self.k, self.lo, self.hi = 1 / fast_tau, 1 / slow_tau, k, lo, hi
        self.fast = self.slow = None
        self.n = 0
        self.gain = 1.0

    def update(self, err2):
        self.n += 1
        if self.fast is None:
            self.fast = self.slow = err2 + 1e-12
        self.fast += self.rf * (err2 - self.fast)
        self.slow += max(self.rs, 1 / self.n) * (err2 - self.slow)
        if self.n > 1 / self.rf:
            self.gain = float(np.clip((self.fast / self.slow) ** self.k, self.lo, self.hi))
        return self.gain


class Brain:
    def __init__(self, n_inputs, heads, n_neurons=256, seed=0,
                 plastic=True, hebbian=True, modulation=True, neurogenesis=True,
                 replace_rate=1e-5, maturity=5000, input_tau=10000.0, input_clip=5.0,
                 modulator_kw=None, sleep_only=False, **core_kw):
        """sleep_only: slow weights, plasticity coefficients and neuron replacement are
        accumulated while awake and applied only in sleep(), so the representation the
        readouts learn from is stable during the day (hippocampus/cortex split)."""
        self.rng = np.random.default_rng(seed)
        self.core = PPCCore(n_inputs, n_neurons, rng=self.rng, **core_kw)
        self.core.track = plastic
        self.core.defer = self.sleep_only = sleep_only
        self.heads = {h.name: h for h in heads}
        assert len(self.heads) == len(heads), "head names must be unique"
        self.plastic, self.hebbian, self.modulation, self.neurogenesis = plastic, hebbian, modulation, neurogenesis
        self.mod = Modulator(**(modulator_kw or {}))
        self.inp = _Scale(n_inputs, tau=input_tau)
        self.input_clip = input_clip
        self.replace_rate, self.maturity = replace_rate, maturity
        N = n_neurons
        self.utility = np.zeros(N)
        self.age = np.zeros(N, dtype=np.int64)
        self._birth_debt = 0.0
        self.t = 0
        self.births = 0

    @property
    def N(self):
        return self.core.N

    # ---- one tick ------------------------------------------------------------
    def step(self, x, signals=None):
        signals = signals or {}
        core, N = self.core, self.core.N
        x = np.asarray(x, float)
        x = np.where(np.isfinite(x), x, self.inp.mu)
        self.inp.update(x)
        xn = np.clip((x - self.inp.mu) / self.inp.sd, -self.input_clip, self.input_clip)

        h = core.sense(xn)
        phi = np.concatenate([h, xn, [1.0]])

        L = np.zeros(N)
        err2, n_err = 0.0, 0
        for name, head in self.heads.items():
            w_core = head.w[:, :N].copy() if (head.feeds_core and head.w is not None and N) else None
            delta = head.learn(signals.get(name), phi)
            if delta is None or w_core is None:
                continue
            L += head.core_weight * (w_core.T @ delta)
            err2 += float(delta @ delta)
            n_err += len(delta)

        gain = self.mod.update(err2 / n_err) if (self.modulation and n_err) else 1.0
        if self.plastic and n_err:
            core.learn(L, gain)
        if self.hebbian:
            core.hebbian(gain)
        if self.neurogenesis and N:
            self._renew(h)

        self.t += 1
        return {name: head.predict(phi) for name, head in self.heads.items()}

    # ---- structural plasticity -----------------------------------------------
    def _renew(self, h):
        N = self.N
        out = np.zeros(N)
        for head in self.heads.values():
            if head.w is not None:
                out += np.abs(head.w[:, :N]).sum(0)
        out += np.abs(self.core.W[:, :N]).sum(0) / max(N, 1)   # also counts if others listen
        self.utility += 0.001 * (np.abs(h - self.core.h_mean) * out - self.utility)
        self.age += 1
        self._birth_debt += self.replace_rate * (self.age > self.maturity).sum()
        if not getattr(self, "sleep_only", False):
            self._births()

    def _births(self):
        """Replace the lowest-utility mature neurons, one per unit of accumulated debt."""
        mature = self.age > self.maturity
        while self._birth_debt >= 1.0 and mature.any():
            self._birth_debt -= 1.0
            cand = np.flatnonzero(mature)
            i = cand[np.argmin(self.utility[cand])]
            self.core.rebirth(i)
            for head in self.heads.values():
                if head.w is not None:
                    head.w[:, i] = 0.0
                if getattr(head, "trace", None) is not None:
                    head.trace[i] = 0.0
                if getattr(head, "P", None) is not None:   # RLS: forget the old neuron's statistics
                    head.P[i, :] = 0.0
                    head.P[:, i] = 0.0
                    head.P[i, i] = 10.0
            self.utility[i] = np.median(self.utility)
            self.age[i] = 0
            mature[i] = False
            self.births += 1

    # ---- sleep -----------------------------------------------------------------
    def sleep(self, frac=0.5):
        """Consolidate fast -> slow weights (call at quiet periods, e.g. market close)."""
        self.core.consolidate(frac)
        if getattr(self, "sleep_only", False):
            self.core.apply_deferred()
            if self.neurogenesis and self.N:
                self._births()

    # ---- persistence -------------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def stats(self):
        c = self.core
        return {"t": self.t, "births": self.births, "gain": self.mod.gain,
                "|W|": float(np.abs(c.W).mean()) if c.N else 0.0,
                "|F|": float(np.abs(c.F).mean()) if c.N else 0.0,
                "A": float(c.A.mean()) if c.N else 0.0}
