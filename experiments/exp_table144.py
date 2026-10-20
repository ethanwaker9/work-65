from common import *


def main():
    data = load_json("waves_F2.json")
    waves = genuine_waves(data["waves"])
    ver = load_json("multistability.json")["verification"]
    prot = load_json("protocol144.json")
    model = Lorenz96(2.0)
    rows = []
    seen = set()
    for w in waves:
        d = w["d"]
        if 144 % d or d in seen:
            continue
        seen.add(d)
        m = 144 // d
        th = 2 * np.pi * np.arange(1, m // 2 + 1) / m
        mus = wave_bloch_multipliers(model, np.array(w["x"])[:, None], w["tau"], w["kappa"], 200, th)
        rho = float(max(np.max(np.abs(mu)) for mu in mus))
        dev = [v["deviation"][-1] for v in ver if v["N"] == 144 and v["d"] == d]
        rows.append({"d": d, "m": m, "l": w["l"], "T": w["T"], "D": w["D"], "rho_roots": rho,
                     "stable": rho < 1.0, "final_dev": dev[0] if dev else None,
                     "seen_random": sum(int(c) for a, cnt in prot.items() for p, c in cnt.items() if int(p) == d)})
        print(rows[-1], flush=True)
    save_json("table144.json", rows)


if __name__ == "__main__":
    main()
