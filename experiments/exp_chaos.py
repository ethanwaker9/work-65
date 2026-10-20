from common import *
from sublattice.models import RosslerRing, LorenzRing, HindmarshRoseRing
from sublattice.bloch import bloch_lyapunov, orbit_bloch_multipliers
from sublattice.waves import periodic_shoot, detect_period, settle
from sublattice.baselines import bqr_lyapunov

RINGS = {
    "Rossler": (RosslerRing, [1.0, 1.0, 0.5], 0.02, 3000.0, 300.0),
    "Lorenz": (LorenzRing, [1.0, 1.0, 20.0], 0.004, 400.0, 50.0),
    "HindmarshRose": (HindmarshRoseRing, [-1.0, -5.0, 3.0], 0.02, 4000.0, 500.0),
}
EPS = {"Rossler": 1.0, "Lorenz": 3.0, "HindmarshRose": 1.0}


def kaplan_yorke(exps):
    e = np.sort(np.asarray(exps))[::-1]
    s = np.cumsum(e)
    if e[0] < 0:
        return 0.0
    j = int(np.max(np.where(s >= 0)[0]))
    if j == e.size - 1:
        return float(e.size)
    return float(j + 1 + s[j] / abs(e[j + 1]))


def msf_curves():
    out = {}
    for name, (cls, x0, h, T, tr) in RINGS.items():
        eps = 4.0 if name == "Lorenz" else 1.5
        model = cls(eps)
        th = np.arccos(1.0 - np.linspace(0.0, 2.0, 61))
        L, _ = bloch_lyapunov(model, np.array([x0]), th, h, int(T / h), 5, int(tr / h))
        out[name] = {"alpha": (eps * (2.0 - 2.0 * np.cos(th))).tolist(), "lam": L.tolist()}
        print(name, "base", np.round(L[0], 4), "top", np.round(L[:, 0], 3), flush=True)
    return out


def ring_sizes(NMAX=40):
    from sublattice.admissible import farey_phases
    out = {}
    fr = [(0, 1)] + farey_phases(NMAX)
    th = np.array([2.0 * np.pi * a / b for a, b in fr])
    for name, (cls, x0, h, T, tr) in RINGS.items():
        eps = EPS[name]
        model = cls(eps)
        L, _ = bloch_lyapunov(model, np.array([x0]), th, h, int(T / h), 5, int(tr / h))
        table = {}
        for (a, b), lam in zip(fr, L):
            table[(a, b)] = lam
        rows = []
        for N in range(2, NMAX + 1):
            full = list(table[(0, 1)])
            trans = -np.inf
            for r in range(1, N // 2 + 1):
                from math import gcd
                g = gcd(r, N)
                lam = table[(r // g, N // g)]
                mult = 1 if (N % 2 == 0 and r == N // 2) else 2
                full += list(lam) * mult
                trans = max(trans, float(lam[0]))
            rows.append({"N": N, "max_transverse": trans, "DKY_full": kaplan_yorke(full),
                         "DKY_base": kaplan_yorke(table[(0, 1)]), "n_full": len(full)})
            print(name, N, round(trans, 4), round(kaplan_yorke(full), 3), round(kaplan_yorke(table[(0, 1)]), 3), flush=True)
        out[name] = {"eps": eps, "rows": rows, "base": list(table[(0, 1)])}
    return out


def rossler_upo_window():
    model = RosslerRing(1.0)
    x = settle(model, np.array([[1.0, 1.0, 0.5]]), h=0.01, t=300.0)
    s = solve_period1(model, x)
    res = {}
    alphas = np.linspace(0.0, 0.6, 61)
    th = np.arccos(1.0 - alphas / 2.0)
    if s is not None:
        xs, T = s
        mus = orbit_bloch_multipliers(model, xs, T, int(np.ceil(T / 0.002)), th)
        lam_upo = [float(np.log(nontrivial_radius(mu)) / T) if i == 0 else float(np.log(np.max(np.abs(mu))) / T)
                   for i, mu in enumerate(mus)]
        res["upo_T"] = T
        res["upo_x"] = xs[0].tolist()
        res["lam_upo"] = lam_upo
    L, _ = bloch_lyapunov(model, np.array([[1.0, 1.0, 0.5]]), th, 0.02, int(4000 / 0.02), 5, int(300 / 0.02))
    res["alpha"] = alphas.tolist()
    res["lam_nat"] = L[:, 0].tolist()
    return res


def solve_period1(model, x):
    from scipy.integrate import solve_ivp
    best = None
    for T0 in np.linspace(5.6, 6.2, 13):
        xs, T, r = periodic_shoot(model, x, T0, 1500)
        if r < 1e-10 and 5.0 < T < 6.5:
            best = (xs, T)
            break
    return best


def l96_saddle():
    F = 8.0
    model = Lorenz96(F)
    rng = np.random.default_rng(3)
    x = settle(model, (F + 0.5 * rng.standard_normal(5))[:, None], h=0.01, t=300.0)
    th = np.linspace(0.0, np.pi, 61)
    L, _ = bloch_lyapunov(model, x, th, 0.01, 60000, 5)
    counts = []
    for m in range(1, 41):
        r = np.arange(m // 2 + 1)
        tt = 2.0 * np.pi * r / m
        Lm, _ = bloch_lyapunov(model, x, tt, 0.01, 30000, 5)
        npos = int(np.sum(Lm[0] > 1e-3))
        for i in range(1, r.size):
            mult = 1 if (m % 2 == 0 and r[i] == m // 2) else 2
            npos += mult * int(np.sum(Lm[i] > 1e-3))
        counts.append({"m": m, "N": 5 * m, "n_positive": npos})
        print("L96 saddle", m, npos, flush=True)
    actual = []
    for m in [2, 4]:
        xx = settle(model, np.tile(x, (m, 1)) + 1e-3 * rng.standard_normal((5 * m, 1)), h=0.01, t=200.0)
        Lf = bqr_lyapunov(model, xx, 1, 0.01, 30000, 5)
        actual.append({"N": 5 * m, "spectrum": Lf.tolist(), "n_positive": int(np.sum(Lf > 1e-3)),
                       "DKY": kaplan_yorke(Lf)})
        print("L96 attractor N", 5 * m, "npos", int(np.sum(Lf > 1e-3)), flush=True)
    return {"theta": th.tolist(), "bands": L.tolist(), "counts": counts, "actual": actual}


def main():
    out = {"msf": msf_curves()}
    save_json("chaos.json", out)
    out["upo"] = rossler_upo_window()
    save_json("chaos.json", out)
    out["rings"] = ring_sizes()
    save_json("chaos.json", out)
    out["l96"] = l96_saddle()
    save_json("chaos.json", out)


if __name__ == "__main__":
    main()
