"""Readout heads: what the brain predicts. Each head learns online from signals that
arrive AFTER its prediction was made, so learning never peeks at the future.

PredictHead  -- "what will signal y be, `delay` ticks from now?"  Target arrives at
                t+delay; the feature vector used at prediction time is kept in a ring
                buffer until then.  kind='regression' (squared loss) or 'binary' (logistic).
GVFHead      -- "how much of cumulant c will I receive, discounted by gamma?"
                (general value function, learned by TD -- the dopamine-style signal).
                Needs no buffer: error is available one tick later, whatever the horizon
                (horizon ~ 1/(1-gamma) ticks).  This is how long horizons learn online.

Both learn in normalised units and report in raw units.  Weights update by NLMS.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class _Scale:
    """Running mean/RMS of a signal (slow EMA, cumulative during warm-up)."""

    def __init__(self, dim, tau=10000.0, center=True):
        self.mu, self.var, self.n = np.zeros(dim), np.ones(dim), 0
        self.tau, self.center = tau, center

    def update(self, y):
        self.n += 1
        r = max(1.0 / self.n, 1.0 / self.tau)
        if self.center:
            d = y - self.mu
            self.mu += r * d
            self.var += r * (d * d - self.var)
        else:
            self.var += r * (y * y - self.var)

    @property
    def sd(self):
        return np.sqrt(self.var) + 1e-12


class PredictHead:
    def __init__(self, name, dim=1, delay=1, kind="regression", lr=0.01, core_weight=1.0,
                 solver="nlms", rls_halflife=200000.0):
        """solver='nlms': one small gradient step per label (slow, robust).
        solver='rls': recursive least squares with exponential forgetting -- each label is used
        about as efficiently as exact regression on all past data, half-weighted after
        `rls_halflife` labels (regression only)."""
        assert kind in ("regression", "binary") and delay >= 1 and solver in ("nlms", "rls")
        assert solver == "nlms" or kind == "regression"
        self.name, self.dim, self.delay, self.kind = name, dim, delay, kind
        self.lr, self.core_weight = lr, core_weight
        self.solver, self.lam = solver, 0.5 ** (1.0 / rls_halflife)
        self.P = None
        self.w = None
        self.buf = deque(maxlen=delay)
        self.scale = _Scale(dim)

    @property
    def feeds_core(self):   # only delay-1 errors line up with the core's eligibility
        return self.delay == 1 and self.core_weight > 0

    def _init(self, P):
        self.w = np.zeros((self.dim, P))
        if getattr(self, "solver", "nlms") == "rls":
            self.P = np.eye(P) * 10.0

    def _out(self, phi):
        u = self.w @ phi
        return 1.0 / (1.0 + np.exp(-u)) if self.kind == "binary" else u

    def learn(self, y, phi_now):
        """y: realised target for the prediction made `delay` ticks ago (None = no label)."""
        if y is None or len(self.buf) < self.delay:
            return None
        y = np.atleast_1d(np.asarray(y, float))
        if np.any(~np.isfinite(y)):
            return None
        phi = self.buf[0]
        if self.kind == "regression":
            self.scale.update(y)
            yn = (y - self.scale.mu) / self.scale.sd
        else:
            yn = y
        delta = yn - self._out(phi)
        if getattr(self, "solver", "nlms") == "rls":
            Pphi = self.P @ phi
            k = Pphi / (self.lam + phi @ Pphi)
            self.w += np.outer(delta, k)
            O = self.__dict__.get("_O")
            if O is None or O.shape != self.P.shape:
                O = self._O = np.empty_like(self.P)     # reused scratch; same arithmetic
            np.outer(k, Pphi, out=O)
            self.P -= O
            self.P /= self.lam
            tr = np.trace(self.P)                 # unexcited directions grow as 1/lam^t: cap them
            if tr > 10.0 * len(phi):
                self.P *= 10.0 * len(phi) / tr
        else:
            self.w += (self.lr / (phi @ phi + 1e-6)) * np.outer(delta, phi)
        return delta

    def predict(self, phi):
        if self.w is None:
            self._init(len(phi))
        self.buf.append(phi)
        p = self._out(phi)
        return self.scale.mu + self.scale.sd * p if self.kind == "regression" else p


class GVFHead:
    def __init__(self, name, gamma, lam=0.0, dim=1, lr=0.1, core_weight=1.0):
        self.name, self.gamma, self.lam, self.dim = name, gamma, lam, dim
        self.lr, self.core_weight = lr, core_weight
        self.w = None
        self.trace = None
        self.phi_prev = None
        self.scale = _Scale(dim, center=False)
        self.pp = 1.0   # running mean of |phi|^2

    feeds_core = property(lambda self: self.core_weight > 0)

    @property
    def horizon(self):
        return 1.0 / (1.0 - self.gamma)

    def learn(self, c, phi_now):
        """c: cumulant received between t-1 and t (e.g. the return of the last tick)."""
        if self.w is None:
            self.w = np.zeros((self.dim, len(phi_now)))
            self.trace = np.zeros(len(phi_now))
        if c is None or self.phi_prev is None:
            self.trace[:] = 0.0
            return None
        c = np.atleast_1d(np.asarray(c, float))
        if np.any(~np.isfinite(c)):
            self.trace[:] = 0.0
            return None
        self.scale.update(c)
        cn = c / self.scale.sd
        phi = self.phi_prev
        delta = cn + self.gamma * (self.w @ phi_now) - self.w @ phi
        self.trace = self.gamma * self.lam * self.trace + phi
        self.pp += 0.001 * (phi @ phi - self.pp)
        self.w += (self.lr * (1 - self.gamma * self.lam) / self.pp) * np.outer(delta, self.trace)
        return delta

    def predict(self, phi):
        if self.w is None:
            self.w = np.zeros((self.dim, len(phi)))
            self.trace = np.zeros(len(phi))
        self.phi_prev = phi
        return self.scale.sd * (self.w @ phi)
