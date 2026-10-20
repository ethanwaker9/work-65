from common import *
from math import gcd
from sublattice.admissible import admissible_multiples

NMAX = 240


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def predict(waves, N, Mfarey):
    out = []
    for w in waves:
        d = w["d"]
        if N % d or d == N:
            continue
        m = N // d
        if m <= Mfarey:
            ok = m in w["admissible"]
        else:
            ok = bool(w["universal_on_grid"])
        if ok:
            out.append({"d": d, "kappa": w["kappa"], "l": w["l"], "T": w["T"]})
    return out


def verify(model, w, N, t=1500.0, amp=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    x = np.tile(np.array(w["x"])[:, None], (N // w["d"], 1)) + amp * rng.standard_normal((N, 1))
    devs = []
    for _ in range(6):
        x = rk4_state(model, x, 0.01, int(round(t / 6 / 0.01)))
        devs.append(float(np.max(np.abs(np.roll(x, -w["d"], axis=0) - x))))
    return devs


def kerin_engler_protocol(model, N, runs, amp, t=1000.0, seed=11):
    rng = np.random.default_rng(seed)
    X = model.F + amp * rng.standard_normal((runs, N, 1))
    X = rk4_state(model, X, 0.01, int(round(t / 0.01)))
    periods = [minimal_spatial_period(X[i], 1e-5) for i in range(runs)]
    vals, cnt = np.unique(periods, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt)}


def main():
    data = load_json("waves_F2.json")
    model = Lorenz96(data["F"])
    waves = genuine_waves(data["waves"])
    Mf = data["Mfarey"]
    table = []
    for N in range(4, NMAX + 1):
        pred = predict(waves, N, Mf)
        table.append({"N": N, "tau": len(divisors(N)), "inherited": pred, "count": len(pred)})
    res = {"table": table}
    save_json("multistability.json", res)
    ver = []
    for w in waves:
        if 144 % w["d"] == 0 and w["d"] < 144:
            dev = verify(model, w, 144)
            ok = (144 // w["d"]) in w["admissible"] if 144 // w["d"] <= Mf else w["universal_on_grid"]
            ver.append({"N": 144, "d": w["d"], "l": w["l"], "predicted_stable": bool(ok), "deviation": dev})
            print("verify N=144 d=", w["d"], ok, ["%.1e" % v for v in dev], flush=True)
    for N in [60, 72, 96, 120]:
        for w in waves:
            if N % w["d"] == 0 and w["d"] < N:
                dev = verify(model, w, N)
                m = N // w["d"]
                ok = m in w["admissible"] if m <= Mf else w["universal_on_grid"]
                ver.append({"N": N, "d": w["d"], "l": w["l"], "predicted_stable": bool(ok), "deviation": dev})
                print("verify N=", N, "d=", w["d"], ok, ["%.1e" % v for v in dev], flush=True)
    res["verification"] = ver
    save_json("multistability.json", res)
    stats = {}
    for amp in [0.01, 1.0]:
        stats[str(amp)] = kerin_engler_protocol(model, 144, 100, amp)
        print("protocol amp", amp, stats[str(amp)], flush=True)
    res["protocol144"] = stats
    save_json("multistability.json", res)


if __name__ == "__main__":
    main()
