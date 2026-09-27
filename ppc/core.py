"""PPC core: a recurrent population of leaky neurons whose synapses learn every tick.

Each synapse (i <- j) carries four numbers:
  W[i,j]    slow weight      -- learned by online gradient (e-prop style, no BPTT)
  F[i,j]    fast weight      -- Hebbian, decays in ~fast_tau ticks (short-term memory)
  A[i,j]    plasticity coeff -- how much the fast weight matters (learned, like Miconi 2018)
  E[i,j]    eligibility      -- "this synapse was recently responsible for neuron i"

Effective weight = W + A*F.  Presynaptic vector z = [h_{t-1}, x_t, 1].

Neuron i:  h_i(t) = (1 - a_i) h_i(t-1) + a_i tanh(sum_j Weff_ij z_j)
a_i is a per-neuron leak drawn log-uniform so the population spans timescales
from 1 tick to tau_max ticks (fast "reflex" cells to slow "mood" cells).
"""
from __future__ import annotations

import numpy as np


def _clip1(x):
    """In-place clip to [-1, 1]; same result as np.clip (NaN included) without its overhead."""
    np.maximum(x, -1.0, out=x)
    np.minimum(x, 1.0, out=x)


class PPCCore:
    def __init__(self, n_inputs, n_neurons=256, tau_max=1000.0, spectral_radius=0.9,
                 input_scale=1.0, fast_tau=600.0, hebb_lr=1e-3, slow_lr=3e-4,
                 plast_lr=1e-4, init_plasticity=0.1, rng=None):
        self.D, self.N = n_inputs, n_neurons
        self.M = self.N + self.D + 1
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tau_max, self.spectral_radius, self.input_scale = tau_max, spectral_radius, input_scale
        self.fast_decay = 1.0 / fast_tau
        self.hebb_lr, self.slow_lr, self.plast_lr = hebb_lr, slow_lr, plast_lr
        self.init_plasticity = init_plasticity
        self.track = True                # maintain eligibility traces (off for frozen cores)

        N, M = self.N, self.M
        self.alpha = np.exp(self.rng.uniform(np.log(1.0 / tau_max), 0.0, N)) if N else np.zeros(0)
        self.W = np.zeros((N, M))
        for i in range(N):
            self.W[i] = self._fresh_row()
        if N:  # scale recurrent block to target spectral radius
            rec = self.W[:, :N]
            rad = np.max(np.abs(np.linalg.eigvals(rec))) or 1.0
            self.W[:, :N] = rec * (spectral_radius / rad)
        self.F = np.zeros((N, M))
        self.A = np.full((N, M), init_plasticity)
        self.E = np.zeros((N, M))        # eligibility at t
        self.E_prev = np.zeros((N, M))   # eligibility at t-1 (what delay-1 errors refer to)
        self.gW = 1e-4                   # running mean-square of slow-weight gradients
        self.gA = 1e-4
        self.h = np.zeros(N)
        self.h_mean = np.zeros(N)
        self.z = np.zeros(M)

    def _fresh_row(self):
        N, D = self.N, self.D
        row = np.zeros(self.M)
        k = max(1, min(N, 16))   # sparse recurrent fan-in
        if N:
            idx = self.rng.choice(N, k, replace=False)
            row[idx] = self.rng.normal(0, 1.0 / np.sqrt(k), k)
        row[N:N + D] = self.rng.normal(0, self.input_scale / np.sqrt(max(D, 1)), D)
        row[-1] = self.rng.normal(0, 0.1)
        return row

    def _bufs(self):
        """Three reusable N x M scratch matrices (lazy: states pickled before they existed
        still load).  Only avoids per-tick allocation; arithmetic and its order are unchanged."""
        b = self.__dict__.get("_B")
        if b is None or b[0].shape != (self.N, self.M):
            b = self._B = [np.empty((self.N, self.M)) for _ in range(3)]
        return b

    # ---- forward -----------------------------------------------------------
    def sense(self, x):
        """Advance one tick with input x (already normalised). Returns new h."""
        N = self.N
        z = np.empty(self.M)
        z[:N], z[N:N + self.D], z[-1] = self.h, x, 1.0
        if N:
            B = self._bufs()[0]
            np.multiply(self.A, self.F, out=B)
            np.add(self.W, B, out=B)                        # = W + A*F
            a = B @ z
            th = np.tanh(a)
            h = (1 - self.alpha) * self.h + self.alpha * th
            if self.track:
                self.E_prev, self.E = self.E, self.E_prev   # reuse buffer
                np.multiply(self.E_prev, (1 - self.alpha)[:, None], out=self.E)
                np.multiply((self.alpha * (1 - th * th))[:, None], z[None, :], out=B)
                self.E += B
            self.h = h
            self.h_mean += 0.001 * (h - self.h_mean)
        self.z = z
        return self.h

    # ---- plasticity ---------------------------------------------------------
    def hebbian(self, gain):
        """Fast weights: decaying Hebbian trace of (centred post) x (pre), with Oja's
        term (-post^2 * F) so each neuron's fast-weight row self-normalises instead of
        saturating."""
        if not self.N:
            return
        post = self.h - self.h_mean
        B1, B2, _ = self._bufs()
        self.F *= (1 - self.fast_decay)
        np.outer(post, self.z, out=B1)
        np.multiply((post * post)[:, None], self.F, out=B2)
        np.subtract(B1, B2, out=B1)
        np.multiply(self.hebb_lr * gain, B1, out=B1)
        self.F += B1
        _clip1(self.F)

    def learn(self, L, gain):
        """Slow-weight + plasticity-coefficient update from learning signal L (per neuron).

        L_i = sum_k w_out[k,i] * delta_k : broadcast error for neuron i, applied to the
        eligibility of the step the error refers to (t-1).

        Step size is normalised by ONE running scale per matrix, not per synapse: per-synapse
        (Adam/RMSprop) normalisation moves every synapse ~lr per tick even when its gradient
        is pure noise, which random-walks the network on low-SNR streams.
        """
        if not self.N:
            return
        g, gA, B = self._bufs()
        np.multiply(L[:, None], self.E_prev, out=g)
        np.multiply(g, g, out=B)
        self.gW += 0.001 * (float(np.mean(B)) - self.gW)
        np.multiply(self.slow_lr * gain / (np.sqrt(self.gW) + 1e-8), g, out=B)
        defer = getattr(self, "defer", False)
        if defer:
            dW, dA = self._deferred()
            dW += B                                     # applied at sleep, not now
        else:
            self.W += B
        np.multiply(g, self.F, out=gA)
        np.multiply(gA, gA, out=B)
        self.gA += 0.001 * (float(np.mean(B)) - self.gA)
        np.multiply(self.plast_lr * gain / (np.sqrt(self.gA) + 1e-8), gA, out=B)
        if defer:
            dA += B
        else:
            self.A += B
            _clip1(self.A)

    def _deferred(self):
        d = self.__dict__.get("_D")
        if d is None or d[0].shape != (self.N, self.M):
            d = self._D = [np.zeros((self.N, self.M)), np.zeros((self.N, self.M))]
        return d

    def apply_deferred(self):
        """Sleep: apply the slow-weight change accumulated while awake."""
        if not self.N:
            return
        dW, dA = self._deferred()
        self.W += dW
        self.A += dA
        _clip1(self.A)
        dW[:] = 0.0
        dA[:] = 0.0

    def consolidate(self, frac):
        """Sleep: move frac of the fast-weight contribution into slow weights.
        W + A*F is unchanged by this call; only what persists after F decays changes."""
        if not self.N:
            return
        self.W += frac * self.A * self.F
        self.F *= (1 - frac)

    def rebirth(self, i):
        """Replace neuron i: fresh input synapses, silenced output synapses."""
        self.W[i] = self._fresh_row()
        self.W[:, i] = 0.0      # nobody listens to the newborn yet
        self.F[i] = 0.0
        self.F[:, i] = 0.0
        self.A[i] = self.init_plasticity
        self.E[i] = self.E_prev[i] = 0.0
        self.h[i] = self.h_mean[i] = 0.0
