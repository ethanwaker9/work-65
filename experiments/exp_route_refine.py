from common import *
from sublattice.waves import settle, detect_period
from sublattice.bloch import orbit_bloch_multipliers


def orbit_at(F, x):
    model = Lorenz96(F)
    x = settle(model, x, h=0.01, t=1500.0)
    det = detect_period(model, x, h=0.001, tmax=60.0, tol=1e-3, nsteps_per_unit=500)
    return model, x, det


def worst(model, xs, T, m):
    th = 2 * np.pi * np.arange(1, m // 2 + 1) / m
    mus = orbit_bloch_multipliers(model, xs, T, max(800, int(np.ceil(T / 0.002))), th)
    rho = max(np.max(np.abs(mu)) for mu in mus)
    return np.log(rho) / T, th[int(np.argmax([np.max(np.abs(mu)) for mu in mus]))]


def main():
    rng = np.random.default_rng(1)
    x0 = (3.9 + 0.3 * rng.standard_normal(5))[:, None]
    _, x0, _ = orbit_at(3.9, x0)
    out = []
    for m in range(2, 13):
        lo, hi = 3.94, 4.06
        x = x0.copy()
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            model, xx, det = orbit_at(mid, x.copy())
            if det is None:
                hi = mid
                continue
            xs, T = det
            eta, thw = worst(model, xs, T, m)
            if eta < 0:
                lo = mid
            else:
                hi = mid
        model, xx, det = orbit_at(hi, x.copy())
        thw = None
        if det is not None:
            xs, T = det
            _, thw = worst(model, xs, T, m)
        out.append({"m": m, "F_star": 0.5 * (lo + hi), "lo": lo, "hi": hi, "theta_first": thw})
        print(m, round(lo, 5), round(hi, 5), thw, flush=True)
    save_json("route_thresholds.json", out)


if __name__ == "__main__":
    main()
