from common import *
from math import gcd, pi
from sublattice.hopf import first_hopf, hopf_inheritance, superlevel_interval, g


def second_hopf(N, FH):
    ls = np.arange(1, N)
    vals = np.sort(np.unique(np.round(g(2.0 * pi * ls / N), 12)))[::-1]
    return 1.0 / vals[1] if vals.size > 1 and vals[1] > 0 else 2.0 * FH


def simulate_onset(N, F, t=20000.0, h=0.05, seed=0):
    rng = np.random.default_rng(seed)
    m = Lorenz96(F)
    x = F + 1e-3 * rng.standard_normal((N, 1))
    x = rk4_state(m, x, h, int(round(t / h)))
    return minimal_spatial_period(x, 1e-5)


def main():
    rows = []
    for N in range(4, 61):
        FH, crit = first_hopf(N)
        entry = {"N": N, "FH": FH, "modes": crit, "hopf_hopf": len(crit) > 1}
        if len(crit) == 1:
            F2 = second_hopf(N, FH)
            entry["F_sim"] = FH + 0.3 * min(F2 - FH, 0.03)
            entry["observed_period"] = simulate_onset(N, entry["F_sim"])
        rows.append(entry)
        print(N, FH, crit, entry.get("observed_period"), flush=True)
    pattern = []
    for m in range(1, 16):
        N = 5 * m
        FH, crit = first_hopf(N)
        pattern.append({"m": m, "N": N, "FH": FH, "modes": crit})
    horizons = []
    for p in range(4, 21):
        r = hopf_inheritance(p, 30)
        if r is None:
            continue
        horizons.append(r)
    save_json("hopf.json", {"rows": rows, "pattern5": pattern, "horizons": horizons})


if __name__ == "__main__":
    main()
