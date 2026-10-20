from common import *
from sublattice.models import RosslerRing
from sublattice.bloch import orbit_bloch_multipliers, bloch_lyapunov
from math import gcd


def main():
    ch = load_json("chaos.json")
    model = RosslerRing(1.0)
    xs = np.array([ch["upo"]["upo_x"]])
    T = ch["upo"]["upo_T"]
    out = {"N": [], "nat": [], "upo": []}
    rings = {r["N"]: r for r in ch["rings"]["Rossler"]["rows"]}
    for N in range(2, 26):
        th = 2 * np.pi * np.arange(1, N // 2 + 1) / N
        mus = orbit_bloch_multipliers(model, xs, T, 3000, th)
        lu = max(np.log(np.max(np.abs(mu))) / T for mu in mus)
        out["N"].append(N)
        out["upo"].append(float(lu))
        out["nat"].append(rings[N]["max_transverse"])
        print(N, round(rings[N]["max_transverse"], 4), round(lu, 4), flush=True)
    th = np.linspace(0, np.pi, 81)
    mus = orbit_bloch_multipliers(model, xs, T, 3000, th)
    out["alpha_full"] = (2 - 2 * np.cos(th)).tolist()
    out["upo_full"] = [float(np.log(np.max(np.abs(mu))) / T) if i else float(np.log(nontrivial_radius(mu)) / T) for i, mu in enumerate(mus)]
    save_json("rossler_window.json", out)


if __name__ == "__main__":
    main()
