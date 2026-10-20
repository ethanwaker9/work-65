import numpy as np
from math import gcd, acos, pi, cos

THETA_STAR = acos(0.25)


def g(theta):
    return np.cos(theta) - np.cos(2.0 * theta)


def h(theta):
    return np.sin(theta) + np.sin(2.0 * theta)


def first_hopf(N, sign=1, tol=1e-12):
    ls = np.arange(1, N)
    th = 2.0 * pi * ls / N
    vals = sign * g(th)
    vmax = vals.max()
    crit = ls[np.abs(vals - vmax) < tol]
    crit = sorted(set(int(min(l, N - l)) for l in crit))
    FH = sign / vmax
    out = []
    for l in crit:
        gg = gcd(l, N)
        complex_pair = abs(h(2.0 * pi * l / N)) > 1e-12
        out.append({"l": l, "gcd": gg, "spatial_period": N // gg, "complex": complex_pair})
    return FH, out


def superlevel_interval(theta_p):
    c0 = cos(theta_p)
    gp = c0 - (2.0 * c0 * c0 - 1.0)
    disc = 1.0 - 8.0 * (gp - 1.0)
    c1 = (1.0 + np.sqrt(disc)) / 4.0
    c2 = (1.0 - np.sqrt(disc)) / 4.0
    t1 = acos(min(1.0, max(-1.0, c1)))
    t2 = acos(min(1.0, max(-1.0, c2)))
    return min(t1, t2), max(t1, t2)


def hopf_inheritance(p, mmax):
    FH, crit = first_hopf(p)
    if len(crit) != 1:
        return None
    lp = crit[0]["l"]
    lo, hi = superlevel_interval(2.0 * pi * lp / p)
    res = []
    for m in range(1, mmax + 1):
        N = p * m
        FHN, critN = first_hopf(N)
        lifted = any(c["l"] == lp * m for c in critN)
        unique = len(critN) == 1
        res.append({"m": m, "N": N, "lifted_first": lifted and unique, "tie": lifted and not unique,
                    "modes": [c["l"] for c in critN]})
    a = lo / (2.0 * pi)
    b = hi / (2.0 * pi)
    return {"p": p, "l": lp, "theta_lo": lo, "theta_hi": hi, "a": a, "b": b, "table": res}
