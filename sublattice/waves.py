import numpy as np
from math import gcd
from .bloch import Stencil, rk4_state, rk4_real_variational


def settle(model, x0, h=0.01, t=500.0):
    return rk4_state(model, np.array(x0, dtype=float), h, int(round(t / h)))


def minimal_spatial_period(x, tol=1e-6):
    d = x.shape[0]
    for p in range(1, d + 1):
        if d % p == 0 and np.max(np.abs(np.roll(x, -p, axis=0) - x)) < tol:
            return p
    return d


def reduce_to_period(x, tol=1e-6):
    p = minimal_spatial_period(x, tol)
    return x[:p].copy()


def detect_rotating_wave(model, x, h=0.002, tmax=60.0, tol=2e-2, tmin=0.02, nsteps=200):
    d = x.shape[0]
    kaps = [kap for kap in range(1, d) if gcd(kap, d) == 1]
    if not kaps:
        return None
    n = int(round(tmax / h))
    scale = max(1.0, np.linalg.norm(x))
    targets = np.stack([np.roll(x, -kap, axis=0) for kap in kaps])
    y = x.copy()
    dp2 = None
    dp = None
    for i in range(1, n + 1):
        y = rk4_state(model, y, h, 1)
        dist = np.linalg.norm((targets - y[None]).reshape(len(kaps), -1), axis=1) / scale
        if dp2 is not None and i * h > tmin:
            cand = np.where((dp < dp2) & (dp <= dist) & (dp < tol))[0]
            for c in cand[np.argsort(dp[cand])]:
                kap = kaps[int(c)]
                xs, tau, res = reduced_shoot(model, x, (i - 1) * h, kap, nsteps)
                if res < 1e-10 and tau > 0:
                    return kap, tau, xs
        dp2, dp = dp, dist
    return None


def reduced_shoot(model, x0, tau0, kappa, nsteps, iters=50, tol=1e-12):
    d, k = x0.shape
    st = Stencil(model.offsets, d)
    x = x0.copy()
    tau = float(tau0)
    E = np.zeros((d, k, d * k))
    for j in range(d):
        for c in range(k):
            E[j, c, j * k + c] = 1.0
    res = np.inf
    for _ in range(iters):
        xt, U = rk4_real_variational(model, x.copy(), E.copy(), tau / nsteps, nsteps, st)
        Phi = U.reshape(d * k, d * k)
        r = (xt - np.roll(x, -kappa, axis=0)).ravel()
        res = np.linalg.norm(r)
        if res < tol:
            break
        P = np.zeros((d * k, d * k))
        for j in range(d):
            jj = (j + kappa) % d
            P[j * k:(j + 1) * k, jj * k:(jj + 1) * k] = np.eye(k)
        ft = model.rhs(xt).ravel()
        f0 = model.rhs(x).ravel()
        A = np.zeros((d * k + 1, d * k + 1))
        A[:d * k, :d * k] = Phi - P
        A[:d * k, d * k] = ft
        A[d * k, :d * k] = f0
        if not np.isfinite(res) or res > 1e2:
            return x, tau, np.inf
        try:
            dz = np.linalg.lstsq(A, np.concatenate([-r, [0.0]]), rcond=None)[0]
        except np.linalg.LinAlgError:
            return x, tau, np.inf
        sc = min(1.0, 0.5 * max(1.0, np.linalg.norm(x)) / max(np.linalg.norm(dz), 1e-300))
        x = x + sc * dz[:d * k].reshape(d, k)
        tau = tau + sc * dz[d * k]
        if tau <= 0:
            return x, tau, np.inf
    return x, tau, res


def periodic_shoot(model, x0, T0, nsteps, iters=50, tol=1e-12):
    d, k = x0.shape
    st = Stencil(model.offsets, d)
    x = x0.copy()
    T = float(T0)
    E = np.zeros((d, k, d * k))
    for j in range(d):
        for c in range(k):
            E[j, c, j * k + c] = 1.0
    res = np.inf
    for _ in range(iters):
        xt, U = rk4_real_variational(model, x.copy(), E.copy(), T / nsteps, nsteps, st)
        Phi = U.reshape(d * k, d * k)
        r = (xt - x).ravel()
        res = np.linalg.norm(r)
        if res < tol:
            break
        A = np.zeros((d * k + 1, d * k + 1))
        A[:d * k, :d * k] = Phi - np.eye(d * k)
        A[:d * k, d * k] = model.rhs(xt).ravel()
        A[d * k, :d * k] = model.rhs(x).ravel()
        if not np.isfinite(res) or res > 1e2:
            return x, T, np.inf
        try:
            dz = np.linalg.lstsq(A, np.concatenate([-r, [0.0]]), rcond=None)[0]
        except np.linalg.LinAlgError:
            return x, T, np.inf
        sc = min(1.0, 0.5 * max(1.0, np.linalg.norm(x)) / max(np.linalg.norm(dz), 1e-300))
        x = x + sc * dz[:d * k].reshape(d, k)
        T = T + sc * dz[d * k]
        if T <= 0:
            return x, T, np.inf
    return x, T, res


def detect_period(model, x, h=0.002, tmax=200.0, tol=1e-2, tmin=0.05, nsteps_per_unit=250):
    n = int(round(tmax / h))
    scale = max(1.0, np.linalg.norm(x))
    y = x.copy()
    dp2 = None
    dp = None
    for i in range(1, n + 1):
        y = rk4_state(model, y, h, 1)
        dist = np.linalg.norm(y - x) / scale
        if dp2 is not None and i * h > tmin and dp < dp2 and dp <= dist and dp < tol:
            T0 = (i - 1) * h
            xs, T, res = periodic_shoot(model, x, T0, max(200, int(np.ceil(T0 * nsteps_per_unit))))
            if res < 1e-9 and T > 0:
                return xs, T
        dp2, dp = dp, dist
    return None


def wave_number(kappa, d):
    for l in range(1, d):
        if (kappa * l) % d == 1:
            return l
    return 0
