//! PPC brain in Rust: the same algorithm as ppc/{core,heads,brain}.py, one native call per tick
//! (`step`) or per day (`run`).  Brains are created by the NumPy code (init + spectral scaling)
//! and converted here losslessly via `from_json`; `to_bytes`/`from_bytes` back pickling.
//!
//! Element-wise arithmetic follows the NumPy operation order.  Dot products and means are
//! summed sequentially (NumPy/BLAS use other orders), so results agree to rounding, not bits.
//! New neurons draw from a small built-in RNG (NumPy's stream cannot be reproduced).

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------- rng (xoshiro256**)
#[derive(Serialize, Deserialize, Clone)]
struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *v = x ^ (x >> 31);
        }
        Rng { s, spare: None }
    }
    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * m);
                return u * m;
            }
        }
    }
    /// k distinct indices from 0..n (partial Fisher-Yates)
    fn choice(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.next_u64() % (n - i) as u64) as usize;
            idx.swap(i, j);
        }
        idx.truncate(k);
        idx
    }
}

// ---------------------------------------------------------------- running scale
#[derive(Serialize, Deserialize, Clone)]
struct Scale {
    mu: Vec<f64>,
    var: Vec<f64>,
    n: u64,
    tau: f64,
    center: bool,
}

impl Scale {
    fn update(&mut self, y: &[f64]) {
        self.n += 1;
        let r = (1.0 / self.n as f64).max(1.0 / self.tau);
        for k in 0..y.len() {
            if self.center {
                let d = y[k] - self.mu[k];
                self.mu[k] += r * d;
                self.var[k] += r * (d * d - self.var[k]);
            } else {
                self.var[k] += r * (y[k] * y[k] - self.var[k]);
            }
        }
    }
    #[inline]
    fn sd(&self, k: usize) -> f64 {
        self.var[k].sqrt() + 1e-12
    }
}

// ---------------------------------------------------------------- core
#[derive(Serialize, Deserialize, Clone)]
struct Core {
    n: usize,
    d: usize,
    m: usize,
    alpha: Vec<f64>,
    w: Vec<f64>,
    f: Vec<f64>,
    a: Vec<f64>,
    e: Vec<f64>,
    e_prev: Vec<f64>,
    gw: f64,
    ga: f64,
    h: Vec<f64>,
    h_mean: Vec<f64>,
    z: Vec<f64>,
    fast_decay: f64,
    hebb_lr: f64,
    slow_lr: f64,
    plast_lr: f64,
    init_plasticity: f64,
    input_scale: f64,
    track: bool,
    defer: bool,
    dw: Vec<f64>,
    da: Vec<f64>,
}


impl Core {
    fn sense(&mut self, x: &[f64]) {
        let (n, d, m) = (self.n, self.d, self.m);
        let mut z = vec![0.0; m];
        z[..n].copy_from_slice(&self.h);
        z[n..n + d].copy_from_slice(x);
        z[m - 1] = 1.0;
        if n > 0 {
            if self.track {
                std::mem::swap(&mut self.e, &mut self.e_prev);
            }
            let mut hn = vec![0.0; n];
            for i in 0..n {
                let r = i * m;
                let acc = dot_eff(&self.w[r..r + m], &self.a[r..r + m], &self.f[r..r + m], &z);
                let th = acc.tanh();
                let al = self.alpha[i];
                hn[i] = (1.0 - al) * self.h[i] + al * th;
                if self.track {
                    let keep = 1.0 - al;
                    let c = al * (1.0 - th * th);
                    for j in 0..m {
                        self.e[r + j] = self.e_prev[r + j] * keep + c * z[j];
                    }
                }
            }
            self.h = hn;
            for i in 0..n {
                self.h_mean[i] += 0.001 * (self.h[i] - self.h_mean[i]);
            }
        }
        self.z = z;
    }

    /// slow-weight + plasticity update (learn), then the Hebbian step, per row (cache-hot).
    /// Order matches Python: learn reads F before the Hebbian step changes it.
    fn plasticity(&mut self, l: Option<&[f64]>, gain: f64, hebbian: bool) {
        let (n, m) = (self.n, self.m);
        if n == 0 {
            return;
        }
        let (mut s_w, mut s_a) = (0.0, 0.0);
        if let Some(l) = l {
            let (mut sg2, mut sga2) = (0.0, 0.0);
            for i in 0..n {
                let r = i * m;
                let (e, f) = (&self.e_prev[r..r + m], &self.f[r..r + m]);
                let li = l[i];
                let (mut a1, mut a2) = ([0.0f64; 8], [0.0f64; 8]);
                let n8 = m / 8 * 8;
                for c in (0..n8).step_by(8) {
                    for q in 0..8 {
                        let g = li * e[c + q];
                        a1[q] += g * g;
                        let ga = g * f[c + q];
                        a2[q] += ga * ga;
                    }
                }
                for j in n8..m {
                    let g = li * e[j];
                    sg2 += g * g;
                    let ga = g * f[j];
                    sga2 += ga * ga;
                }
                sg2 += a1.iter().sum::<f64>();
                sga2 += a2.iter().sum::<f64>();
            }
            let cnt = (n * m) as f64;
            self.gw += 0.001 * (sg2 / cnt - self.gw);
            s_w = self.slow_lr * gain / (self.gw.sqrt() + 1e-8);
            self.ga += 0.001 * (sga2 / cnt - self.ga);
            s_a = self.plast_lr * gain / (self.ga.sqrt() + 1e-8);
        }
        let keep = 1.0 - self.fast_decay;
        let hs = self.hebb_lr * gain;
        let z = &self.z;
        for i in 0..n {
            let r = i * m;
            if let Some(l) = l {
                let li = l[i];
                let e = &self.e_prev[r..r + m];
                if self.defer {
                    let f = &self.f[r..r + m];
                    let (dw, da) = (&mut self.dw[r..r + m], &mut self.da[r..r + m]);
                    for j in 0..m {
                        let g = li * e[j];
                        dw[j] += s_w * g;
                        da[j] += s_a * (g * f[j]);
                    }
                } else {
                    let f = &self.f[r..r + m];
                    let (w, a) = (&mut self.w[r..r + m], &mut self.a[r..r + m]);
                    for j in 0..m {
                        let g = li * e[j];
                        w[j] += s_w * g;
                        a[j] = clamp1(a[j] + s_a * (g * f[j]));
                    }
                }
            }
            if hebbian {
                let post = self.h[i] - self.h_mean[i];
                let post2 = post * post;
                let f = &mut self.f[r..r + m];
                for j in 0..m {
                    let fk = f[j] * keep;
                    f[j] = clamp1(fk + hs * (post * z[j] - post2 * fk));
                }
            }
        }
    }

    fn consolidate(&mut self, frac: f64) {
        for k in 0..self.w.len() {
            self.w[k] += frac * self.a[k] * self.f[k];
            self.f[k] *= 1.0 - frac;
        }
    }

    fn apply_deferred(&mut self) {
        if self.n == 0 || self.dw.is_empty() {
            return;
        }
        for k in 0..self.w.len() {
            self.w[k] += self.dw[k];
            self.a[k] = clamp1(self.a[k] + self.da[k]);
            self.dw[k] = 0.0;
            self.da[k] = 0.0;
        }
    }

    fn rebirth(&mut self, i: usize, rng: &mut Rng) {
        let (n, d, m) = (self.n, self.d, self.m);
        let r = i * m;
        for j in 0..m {
            self.w[r + j] = 0.0;
        }
        let k = n.min(16).max(1);
        let sd = 1.0 / (k as f64).sqrt();
        for idx in rng.choice(n, k) {
            self.w[r + idx] = rng.normal() * sd;
        }
        let sdx = self.input_scale / (d.max(1) as f64).sqrt();
        for j in 0..d {
            self.w[r + n + j] = rng.normal() * sdx;
        }
        self.w[r + m - 1] = rng.normal() * 0.1;
        for row in 0..n {
            self.w[row * m + i] = 0.0;
            self.f[row * m + i] = 0.0;
        }
        for j in 0..m {
            self.f[r + j] = 0.0;
            self.a[r + j] = self.init_plasticity;
            self.e[r + j] = 0.0;
            self.e_prev[r + j] = 0.0;
        }
        self.h[i] = 0.0;
        self.h_mean[i] = 0.0;
    }
}

// ---------------------------------------------------------------- heads
#[derive(Serialize, Deserialize, Clone)]
struct Predict {
    name: String,
    dim: usize,
    delay: usize,
    binary: bool,
    lr: f64,
    core_weight: f64,
    rls: bool,
    lam: f64,
    p: Option<Vec<f64>>,
    w: Option<Vec<f64>>,
    buf: VecDeque<Vec<f64>>,
    scale: Scale,
}

#[derive(Serialize, Deserialize, Clone)]
struct Gvf {
    name: String,
    gamma: f64,
    lam: f64,
    dim: usize,
    lr: f64,
    core_weight: f64,
    w: Option<Vec<f64>>,
    trace: Option<Vec<f64>>,
    phi_prev: Option<Vec<f64>>,
    scale: Scale,
    pp: f64,
}

#[derive(Serialize, Deserialize, Clone)]
enum Head {
    Predict(Predict),
    Gvf(Gvf),
}

fn matvec(w: &[f64], dim: usize, phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    (0..dim).map(|k| dot(&w[k * p..(k + 1) * p], phi)).collect()
}

/// Dot product with 8 independent partial sums so LLVM can vectorise it (like BLAS, the
/// summation order differs from a naive loop: results agree to rounding).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut t = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

/// sum_j (w_j + a_j * f_j) * z_j, vectorisable
#[inline]
fn dot_eff(w: &[f64], a: &[f64], f: &[f64], z: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let n8 = z.len() / 8 * 8;
    for c in (0..n8).step_by(8) {
        for l in 0..8 {
            let k = c + l;
            acc[l] += (w[k] + a[k] * f[k]) * z[k];
        }
    }
    let mut t = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for k in n8..z.len() {
        t += (w[k] + a[k] * f[k]) * z[k];
    }
    t
}

#[inline]
fn clamp1(v: f64) -> f64 {
    v.max(-1.0).min(1.0)
}

impl Head {
    fn name(&self) -> &str {
        match self {
            Head::Predict(h) => &h.name,
            Head::Gvf(h) => &h.name,
        }
    }
    fn dim(&self) -> usize {
        match self {
            Head::Predict(h) => h.dim,
            Head::Gvf(h) => h.dim,
        }
    }
    fn core_weight(&self) -> f64 {
        match self {
            Head::Predict(h) => h.core_weight,
            Head::Gvf(h) => h.core_weight,
        }
    }
    fn feeds_core(&self) -> bool {
        match self {
            Head::Predict(h) => h.delay == 1 && h.core_weight > 0.0,
            Head::Gvf(h) => h.core_weight > 0.0,
        }
    }
    fn w(&self) -> Option<&Vec<f64>> {
        match self {
            Head::Predict(h) => h.w.as_ref(),
            Head::Gvf(h) => h.w.as_ref(),
        }
    }

    fn learn(&mut self, y: Option<&[f64]>, phi_now: &[f64]) -> Option<Vec<f64>> {
        match self {
            Head::Predict(h) => {
                let y = y?;
                if h.buf.len() < h.delay || y.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                let phi = h.buf.front().unwrap().clone();
                let yn: Vec<f64> = if h.binary {
                    y.to_vec()
                } else {
                    h.scale.update(y);
                    (0..h.dim).map(|k| (y[k] - h.scale.mu[k]) / h.scale.sd(k)).collect()
                };
                let w = h.w.as_mut().unwrap();
                let p = phi.len();
                let u = matvec(w, h.dim, &phi);
                let delta: Vec<f64> = (0..h.dim)
                    .map(|k| yn[k] - if h.binary { 1.0 / (1.0 + (-u[k]).exp()) } else { u[k] })
                    .collect();
                if h.rls {
                    let pm = h.p.as_mut().unwrap();
                    let pphi: Vec<f64> = (0..p).map(|i| dot(&pm[i * p..(i + 1) * p], &phi)).collect();
                    let den = h.lam + dot(&phi, &pphi);
                    let kv: Vec<f64> = pphi.iter().map(|v| v / den).collect();
                    for k in 0..h.dim {
                        for j in 0..p {
                            w[k * p + j] += delta[k] * kv[j];
                        }
                    }
                    let inv = 1.0 / h.lam;
                    let mut tr = 0.0;
                    for i in 0..p {
                        let r = i * p;
                        let ki = kv[i];
                        let row = &mut pm[r..r + p];
                        for j in 0..p {
                            row[j] = (row[j] - ki * pphi[j]) * inv;
                        }
                        tr += row[i];
                    }
                    let cap = 10.0 * p as f64;
                    if tr > cap {
                        let s = cap / tr;
                        pm.iter_mut().for_each(|v| *v *= s);
                    }
                } else {
                    let s = h.lr / (dot(&phi, &phi) + 1e-6);
                    for k in 0..h.dim {
                        for j in 0..p {
                            w[k * p + j] += s * (delta[k] * phi[j]);
                        }
                    }
                }
                Some(delta)
            }
            Head::Gvf(h) => {
                let p = phi_now.len();
                if h.w.is_none() {
                    h.w = Some(vec![0.0; h.dim * p]);
                    h.trace = Some(vec![0.0; p]);
                }
                let bad = match y {
                    None => true,
                    Some(c) => c.iter().any(|v| !v.is_finite()),
                };
                if bad || h.phi_prev.is_none() {
                    h.trace.as_mut().unwrap().iter_mut().for_each(|v| *v = 0.0);
                    return None;
                }
                let c = y.unwrap();
                h.scale.update(c);
                let phi = h.phi_prev.as_ref().unwrap();
                let w = h.w.as_mut().unwrap();
                let vn = matvec(w, h.dim, phi_now);
                let vp = matvec(w, h.dim, phi);
                let delta: Vec<f64> = (0..h.dim)
                    .map(|k| c[k] / h.scale.sd(k) + h.gamma * vn[k] - vp[k])
                    .collect();
                let gl = h.gamma * h.lam;
                let tr = h.trace.as_mut().unwrap();
                for j in 0..p {
                    tr[j] = gl * tr[j] + phi[j];
                }
                h.pp += 0.001 * (dot(phi, phi) - h.pp);
                let s = h.lr * (1.0 - h.gamma * h.lam) / h.pp;
                for k in 0..h.dim {
                    for j in 0..p {
                        w[k * p + j] += s * (delta[k] * tr[j]);
                    }
                }
                Some(delta)
            }
        }
    }

    fn predict(&mut self, phi: &[f64]) -> Vec<f64> {
        let p = phi.len();
        match self {
            Head::Predict(h) => {
                if h.w.is_none() {
                    h.w = Some(vec![0.0; h.dim * p]);
                    if h.rls {
                        let mut pm = vec![0.0; p * p];
                        for i in 0..p {
                            pm[i * p + i] = 10.0;
                        }
                        h.p = Some(pm);
                    }
                }
                h.buf.push_back(phi.to_vec());
                if h.buf.len() > h.delay {
                    h.buf.pop_front();
                }
                let u = matvec(h.w.as_ref().unwrap(), h.dim, phi);
                (0..h.dim)
                    .map(|k| {
                        if h.binary {
                            1.0 / (1.0 + (-u[k]).exp())
                        } else {
                            h.scale.mu[k] + h.scale.sd(k) * u[k]
                        }
                    })
                    .collect()
            }
            Head::Gvf(h) => {
                if h.w.is_none() {
                    h.w = Some(vec![0.0; h.dim * p]);
                    h.trace = Some(vec![0.0; p]);
                }
                h.phi_prev = Some(phi.to_vec());
                let u = matvec(h.w.as_ref().unwrap(), h.dim, phi);
                (0..h.dim).map(|k| h.scale.sd(k) * u[k]).collect()
            }
        }
    }
}

// ---------------------------------------------------------------- modulator
#[derive(Serialize, Deserialize, Clone)]
struct Modulator {
    rf: f64,
    rs: f64,
    k: f64,
    lo: f64,
    hi: f64,
    fast: Option<f64>,
    slow: Option<f64>,
    n: u64,
    gain: f64,
}

impl Modulator {
    fn update(&mut self, err2: f64) -> f64 {
        self.n += 1;
        if self.fast.is_none() {
            self.fast = Some(err2 + 1e-12);
            self.slow = Some(err2 + 1e-12);
        }
        let mut f = self.fast.unwrap();
        let mut s = self.slow.unwrap();
        f += self.rf * (err2 - f);
        s += self.rs.max(1.0 / self.n as f64) * (err2 - s);
        self.fast = Some(f);
        self.slow = Some(s);
        if self.n as f64 > 1.0 / self.rf {
            self.gain = (f / s).powf(self.k).max(self.lo).min(self.hi);
        }
        self.gain
    }
}

// ---------------------------------------------------------------- brain
#[derive(Serialize, Deserialize, Clone)]
struct Brain {
    core: Core,
    heads: Vec<Head>,
    modulator: Modulator,
    inp: Scale,
    input_clip: f64,
    plastic: bool,
    hebbian: bool,
    modulation: bool,
    neurogenesis: bool,
    sleep_only: bool,
    replace_rate: f64,
    maturity: i64,
    utility: Vec<f64>,
    age: Vec<i64>,
    birth_debt: f64,
    t: u64,
    births: u64,
    #[serde(default = "default_rng")]
    rng: Rng,
}

fn default_rng() -> Rng {
    Rng::new(0)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

impl Brain {
    fn step(&mut self, x: &[f64], signals: &[Option<&[f64]>]) -> Vec<Vec<f64>> {
        let n = self.core.n;
        let d = self.core.d;
        let mut xn = vec![0.0; d];
        let xs: Vec<f64> = (0..d).map(|k| if x[k].is_finite() { x[k] } else { self.inp.mu[k] }).collect();
        self.inp.update(&xs);
        for k in 0..d {
            let v = (xs[k] - self.inp.mu[k]) / self.inp.sd(k);
            xn[k] = v.max(-self.input_clip).min(self.input_clip);
        }
        self.core.sense(&xn);
        let mut phi = Vec::with_capacity(n + d + 1);
        phi.extend_from_slice(&self.core.h);
        phi.extend_from_slice(&xn);
        phi.push(1.0);

        let mut l = vec![0.0; n];
        let (mut err2, mut n_err) = (0.0, 0usize);
        for (hi, head) in self.heads.iter_mut().enumerate() {
            let w_core: Option<Vec<f64>> = if head.feeds_core() && n > 0 {
                head.w().map(|w| {
                    let p = phi.len();
                    let dim = head.dim();
                    let mut c = vec![0.0; dim * n];
                    for k in 0..dim {
                        c[k * n..(k + 1) * n].copy_from_slice(&w[k * p..k * p + n]);
                    }
                    c
                })
            } else {
                None
            };
            let delta = head.learn(signals[hi], &phi);
            if let (Some(delta), Some(wc)) = (delta, w_core) {
                let cw = head.core_weight();
                for i in 0..n {
                    let mut v = 0.0;
                    for k in 0..delta.len() {
                        v += wc[k * n + i] * delta[k];
                    }
                    l[i] += cw * v;
                }
                err2 += dot(&delta, &delta);
                n_err += delta.len();
            }
        }
        let gain = if self.modulation && n_err > 0 { self.modulator.update(err2 / n_err as f64) } else { 1.0 };
        let learn = self.plastic && n_err > 0;
        if learn || self.hebbian {
            self.core.plasticity(if learn { Some(&l) } else { None }, gain, self.hebbian);
        }
        if self.neurogenesis && n > 0 {
            self.renew();
        }
        self.t += 1;
        self.heads.iter_mut().map(|h| h.predict(&phi)).collect()
    }

    fn renew(&mut self) {
        let n = self.core.n;
        let m = self.core.m;
        let mut out = vec![0.0; n];
        for head in &self.heads {
            if let Some(w) = head.w() {
                let p = m;
                for k in 0..head.dim() {
                    for j in 0..n {
                        out[j] += w[k * p + j].abs();
                    }
                }
            }
        }
        let mut col = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                col[j] += self.core.w[i * m + j].abs();
            }
        }
        for j in 0..n {
            out[j] += col[j] / n.max(1) as f64;
        }
        let mut mature = 0usize;
        for i in 0..n {
            let act = (self.core.h[i] - self.core.h_mean[i]).abs();
            self.utility[i] += 0.001 * (act * out[i] - self.utility[i]);
            self.age[i] += 1;
            if self.age[i] > self.maturity {
                mature += 1;
            }
        }
        self.birth_debt += self.replace_rate * mature as f64;
        if !self.sleep_only {
            self.births_now();
        }
    }

    fn births_now(&mut self) {
        let n = self.core.n;
        let p = self.core.m;
        let mut mature: Vec<bool> = self.age.iter().map(|a| *a > self.maturity).collect();
        while self.birth_debt >= 1.0 && mature.iter().any(|m| *m) {
            self.birth_debt -= 1.0;
            let mut best = usize::MAX;
            for i in 0..n {
                if mature[i] && (best == usize::MAX || self.utility[i] < self.utility[best]) {
                    best = i;
                }
            }
            let i = best;
            self.core.rebirth(i, &mut self.rng);
            for head in self.heads.iter_mut() {
                match head {
                    Head::Predict(h) => {
                        if let Some(w) = h.w.as_mut() {
                            for k in 0..h.dim {
                                w[k * p + i] = 0.0;
                            }
                        }
                        if let Some(pm) = h.p.as_mut() {
                            for j in 0..p {
                                pm[i * p + j] = 0.0;
                                pm[j * p + i] = 0.0;
                            }
                            pm[i * p + i] = 10.0;
                        }
                    }
                    Head::Gvf(h) => {
                        if let Some(w) = h.w.as_mut() {
                            for k in 0..h.dim {
                                w[k * p + i] = 0.0;
                            }
                        }
                        if let Some(tr) = h.trace.as_mut() {
                            tr[i] = 0.0;
                        }
                    }
                }
            }
            self.utility[i] = median(&self.utility);
            self.age[i] = 0;
            mature[i] = false;
            self.births += 1;
        }
    }

    fn sleep(&mut self, frac: f64) {
        if self.core.n > 0 {
            self.core.consolidate(frac);
        }
        if self.sleep_only {
            self.core.apply_deferred();
            if self.neurogenesis && self.core.n > 0 {
                self.births_now();
            }
        }
    }
}

// ---------------------------------------------------------------- python
#[pyclass(module = "ppc.ppc_rs", name = "Brain")]
struct PyBrain {
    b: Brain,
}

#[pymethods]
impl PyBrain {
    /// Build from the JSON state exported by ppc/rust.py (exact float round-trip); `seed` starts
    /// the RNG used for new neurons (its state is saved with the brain afterwards).
    #[staticmethod]
    #[pyo3(signature = (s, seed=0))]
    fn from_json(s: &str, seed: u64) -> PyResult<Self> {
        let mut b: Brain = serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
        b.rng = Rng::new(seed);
        if b.core.defer && b.core.dw.len() != b.core.w.len() {
            return Err(PyValueError::new_err("sleep-only brain needs deferred buffers"));
        }
        Ok(PyBrain { b })
    }
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let b: Brain = bincode::deserialize(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBrain { b })
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let v = bincode::serialize(&self.b).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &v))
    }
    fn head_names(&self) -> Vec<String> {
        self.b.heads.iter().map(|h| h.name().to_string()).collect()
    }
    fn head_dims(&self) -> Vec<usize> {
        self.b.heads.iter().map(|h| h.dim()).collect()
    }
    #[getter]
    fn n(&self) -> usize {
        self.b.core.n
    }
    #[getter]
    fn births(&self) -> u64 {
        self.b.births
    }
    #[getter]
    fn gain(&self) -> f64 {
        self.b.modulator.gain
    }
    #[getter]
    fn t(&self) -> u64 {
        self.b.t
    }
    fn stats(&self) -> (u64, u64, f64, f64, f64, f64) {
        let c = &self.b.core;
        let mean_abs = |v: &Vec<f64>| if v.is_empty() { 0.0 } else { v.iter().map(|x| x.abs()).sum::<f64>() / v.len() as f64 };
        let a = if c.a.is_empty() { 0.0 } else { c.a.iter().sum::<f64>() / c.a.len() as f64 };
        (self.b.t, self.b.births, self.b.modulator.gain, mean_abs(&c.w), mean_abs(&c.f), a)
    }
    /// Matrices for inspection/verification: name -> flat copy.
    fn get(&self, what: &str) -> PyResult<Vec<f64>> {
        let c = &self.b.core;
        Ok(match what {
            "W" => c.w.clone(),
            "F" => c.f.clone(),
            "A" => c.a.clone(),
            "E" => c.e.clone(),
            "h" => c.h.clone(),
            _ => return Err(PyValueError::new_err("unknown field")),
        })
    }
    fn sleep(&mut self, frac: f64) {
        self.b.sleep(frac)
    }
    /// One tick. signals: one entry per head (None = no label this tick).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        signals: Vec<Option<PyReadonlyArray1<'py, f64>>>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        if signals.len() != self.b.heads.len() {
            return Err(PyValueError::new_err("need one signal entry per head"));
        }
        let xv = x.as_slice()?.to_vec();
        if xv.len() != self.b.core.d {
            return Err(PyValueError::new_err("x has the wrong length"));
        }
        let owned: Vec<Option<Vec<f64>>> = signals
            .iter()
            .map(|s| s.as_ref().map(|a| a.as_slice().map(|v| v.to_vec())).transpose())
            .collect::<Result<_, _>>()?;
        let refs: Vec<Option<&[f64]>> = owned.iter().map(|o| o.as_deref()).collect();
        let out = self.b.step(&xv, &refs);
        Ok(out.into_iter().map(|v| PyArray1::from_vec(py, v)).collect())
    }
    /// A whole block of ticks in one call. x [T, D]; signals: per head [T, dim] (NaN row = no
    /// label) or None.  Returns per head predictions [T, dim].  Releases the GIL.
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        signals: Vec<Option<PyReadonlyArray2<'py, f64>>>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let nh = self.b.heads.len();
        if signals.len() != nh {
            return Err(PyValueError::new_err("need one signal entry per head"));
        }
        let xa = x.as_array();
        let (t_len, d) = (xa.shape()[0], xa.shape()[1]);
        if d != self.b.core.d {
            return Err(PyValueError::new_err("x has the wrong width"));
        }
        let xs: Vec<f64> = xa.iter().cloned().collect();
        let dims: Vec<usize> = self.b.heads.iter().map(|h| h.dim()).collect();
        let mut sigs: Vec<Option<Vec<f64>>> = Vec::with_capacity(nh);
        for (k, s) in signals.iter().enumerate() {
            sigs.push(match s {
                None => None,
                Some(a) => {
                    let a = a.as_array();
                    if a.shape()[0] != t_len || a.shape()[1] != dims[k] {
                        return Err(PyValueError::new_err(format!("signal {k} has the wrong shape")));
                    }
                    Some(a.iter().cloned().collect())
                }
            });
        }
        let b = &mut self.b;
        let preds: Vec<Vec<f64>> = py.allow_threads(|| {
            let mut out: Vec<Vec<f64>> = dims.iter().map(|dm| Vec::with_capacity(t_len * dm)).collect();
            for t in 0..t_len {
                let row: Vec<Option<&[f64]>> = (0..nh)
                    .map(|k| match &sigs[k] {
                        None => None,
                        Some(v) => {
                            let s = &v[t * dims[k]..(t + 1) * dims[k]];
                            if s.iter().any(|x| !x.is_finite()) { None } else { Some(s) }
                        }
                    })
                    .collect();
                let p = b.step(&xs[t * d..(t + 1) * d], &row);
                for k in 0..nh {
                    out[k].extend_from_slice(&p[k]);
                }
            }
            out
        });
        preds
            .into_iter()
            .zip(dims)
            .map(|(v, dm)| {
                PyArray1::from_vec(py, v).reshape([t_len, dm]).map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .collect()
    }
}

#[pymodule]
fn ppc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBrain>()?;
    Ok(())
}
