import numpy as np
from math import gcd, pi
from fractions import Fraction


def farey_phases(M):
    out = []
    for b in range(2, M + 1):
        for a in range(1, b // 2 + 1):
            if gcd(a, b) == 1:
                out.append((a, b))
    return out


def bad_denominators(symbol_many, M, tol=0.0):
    fr = farey_phases(M)
    th = np.array([2.0 * pi * a / b for a, b in fr])
    r = symbol_many(th)
    values = {f: float(v) for f, v in zip(fr, r)}
    bad = set(b for (a, b), v in values.items() if v >= 1.0 - tol)
    return bad, values


def admissible_multiples(bad, M):
    out = []
    for m in range(1, M + 1):
        if not any(m % b == 0 for b in bad if b <= m):
            out.append(m)
    return out


def minimal_bad(bad):
    s = sorted(bad)
    out = []
    for b in s:
        if not any(b % c == 0 for c in out):
            out.append(b)
    return out


def simplest_fraction(alpha, beta):
    alpha = Fraction(alpha).limit_denominator(10 ** 12)
    beta = Fraction(beta).limit_denominator(10 ** 12)
    lo_n, lo_d, hi_n, hi_d = 0, 1, 1, 0
    while True:
        mn, md = lo_n + hi_n, lo_d + hi_d
        f = Fraction(mn, md)
        if f <= alpha:
            lo_n, lo_d = mn, md
        elif f >= beta:
            hi_n, hi_d = mn, md
        else:
            return mn, md


def interval_admissible(alpha, beta, m):
    lo = int(np.floor(alpha * m)) + 1
    return not (lo < beta * m)
