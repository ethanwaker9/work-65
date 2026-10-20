import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import time
import tracemalloc
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

from sublattice.models import Lorenz96
from sublattice.bloch import rk4_state, wave_bloch_multipliers, Stencil
from sublattice.waves import detect_rotating_wave, minimal_spatial_period, wave_number


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return super().default(o)


def save_json(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=1)


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def measure(fn, repeats=1):
    best_t = np.inf
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best_t = min(best_t, time.perf_counter() - t0)
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, best_t, peak


def genuine_waves(waves, tmax=6.0):
    return [w for w in waves if w["T"] < tmax and w["rho0"] < 1.0]


def nontrivial_radius(mu):
    mu = np.asarray(mu).ravel()
    i = int(np.argmin(np.abs(mu - 1.0)))
    return float(np.max(np.abs(np.delete(mu, i))))


def same_orbit(model, xa, xb, tau, h=0.002, tol=1e-6):
    d = xa.shape[0]
    n = int(np.ceil(tau / h))
    y = xa.copy()
    best = np.inf
    for i in range(n + 1):
        for s in range(d):
            best = min(best, np.max(np.abs(np.roll(y, -s, axis=0) - xb)))
        y = rk4_state(model, y, h, 1)
    return best < 1e-3


def find_waves(model, d, seeds, t_settle=600.0, h=0.01, amp=0.5, K=200):
    F = model.F
    rng = np.random.default_rng(1000 + d)
    X = F + amp * rng.standard_normal((seeds, d, 1))
    X = rk4_state(model, X, h, int(round(t_settle / h)))
    waves = []
    for s in range(seeds):
        x = X[s]
        if minimal_spatial_period(x, 1e-6) < d:
            continue
        det = detect_rotating_wave(model, x, nsteps=K)
        if det is None:
            continue
        kap, tau, xs = det
        if minimal_spatial_period(xs, 1e-8) < d:
            continue
        dup = False
        for w in waves:
            if abs(w["tau"] - tau) < 1e-7 and w["kappa"] == kap:
                dup = True
                break
        if dup:
            continue
        mu0 = wave_bloch_multipliers(model, xs, tau, kap, K, np.array([0.0]))[0]
        waves.append({"d": d, "kappa": kap, "l": wave_number(kap, d), "tau": tau, "T": d * tau,
                      "x": xs[:, 0].copy(), "rho0": nontrivial_radius(mu0)})
    return waves
