from common import *
from math import gcd
from sublattice.bloch import bloch_lyapunov
from sublattice.waves import settle
from sublattice.admissible import farey_phases
from sublattice.baselines import bqr_lyapunov
from exp_chaos import kaplan_yorke

THR = 0.01


def main():
    F = 8.0
    model = Lorenz96(F)
    rng = np.random.default_rng(3)
    x = settle(model, (F + 0.5 * rng.standard_normal(5))[:, None], h=0.01, t=300.0)
    fr = [(0, 1)] + farey_phases(40)
    th = np.array([2 * np.pi * a / b for a, b in fr])
    L, _ = bloch_lyapunov(model, x, th, 0.01, 60000, 5)
    table = {f: L[i] for i, f in enumerate(fr)}
    counts = []
    for m in range(1, 41):
        n = 0
        for r in range(m):
            g = gcd(r, m)
            a, b = (r // g, m // g) if r else (0, 1)
            if a > b / 2:
                a = b - a
            n += int(np.sum(table[(a, b)] > THR))
        counts.append({"m": m, "n_positive": n})
    ms = np.array([c["m"] for c in counts])
    ns = np.array([c["n_positive"] for c in counts])
    slope = float(np.sum(ms * ns) / np.sum(ms * ms))
    actual = []
    for m in [2, 4, 8]:
        xx = settle(model, np.tile(x, (m, 1)) + 1e-3 * rng.standard_normal((5 * m, 1)), h=0.01, t=300.0)
        Lf = bqr_lyapunov(model, xx, 1, 0.01, 60000, 5)
        actual.append({"N": 5 * m, "n_positive": int(np.sum(Lf > THR)), "DKY": kaplan_yorke(Lf), "top": float(Lf[0])})
        print("attractor N", 5 * m, actual[-1], flush=True)
    base = table[(0, 1)]
    save_json("saddle.json", {"base": base, "counts": counts, "slope": slope, "actual": actual,
                              "theta": th, "lam1": [float(table[f][0]) for f in fr]})
    print("base", np.round(base, 4), "slope", slope, flush=True)
    print([(c["m"], c["n_positive"]) for c in counts], flush=True)


if __name__ == "__main__":
    main()
