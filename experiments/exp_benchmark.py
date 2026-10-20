from common import *
from sublattice.baselines import (sbr_wave, fm_wave, rm_wave, irr_wave, sbd_wave, nk_wave, hill_wave,
                                  sbr_lyapunov, bqr_lyapunov, cqr_lyapunov, irr_lyapunov, sbd_lyapunov, lift)
from sublattice.waves import reduced_shoot, periodic_shoot, settle

WAVE_LIMITS = {"SBR": 2304, "RM": 2304, "FM": 2304, "IRR": 576, "SBD": 1152, "NK": 576, "HILL": 144}
LYAP_LIMITS = {"SBR": 1280, "BQR": 640, "CQR": 160, "IRR": 320, "SBD": 320}


def wave_bench():
    data = load_json("waves_F2.json")
    w = [v for v in data["waves"] if v["d"] == 9][0]
    model = Lorenz96(2.0)
    xs = np.array(w["x"])[:, None]
    d, tau, kap = 9, w["tau"], w["kappa"]
    T = d * tau
    K = 100
    ref = {}
    rows = load_json("bench_wave.json") if os.path.exists(os.path.join(RESULTS, "bench_wave.json")) else []
    done = {(r["method"], r["N"]) for r in rows}
    for r in rows:
        if r["method"] == "SBR":
            ref[r["N"]] = r["rho"]
    for m in [16, 32, 64, 128, 256]:
        N = d * m
        fns = {
            "SBR": lambda: sbr_wave(model, xs, tau, kap, m, K),
            "RM": lambda: rm_wave(model, xs, tau, kap, m, K),
            "FM": lambda: fm_wave(model, xs, T, m, d * K),
            "IRR": lambda: irr_wave(model, xs, T, m, d * K),
            "SBD": lambda: sbd_wave(model, xs, T, m, d * K),
            "NK": lambda: nk_wave(model, xs, T, m, d * K, nev=6),
            "HILL": lambda: hill_wave(model, xs, T, m, nharm=12),
        }
        for name, fn in fns.items():
            if N > WAVE_LIMITS[name] or (name, N) in done:
                continue
            mu, t, mem = measure(fn, repeats=3 if N <= 288 else 1)
            mu = np.asarray(mu).ravel()
            rho = nontrivial_radius(mu)
            if name == "SBR":
                ref[N] = rho
            rows.append({"method": name, "N": N, "m": m, "time": t, "memory": mem, "rho": rho,
                         "err": abs(rho - ref[N]) / ref[N], "n_mult": int(mu.size)})
            print(f"wave N={N:5d} {name:5s} t={t:9.4f}s mem={mem / 2 ** 20:10.3f}MB rho={rho:.12f}", flush=True)
            save_json("bench_wave.json", rows)
    return rows


def lyap_bench():
    model = Lorenz96(8.0)
    rng = np.random.default_rng(3)
    x = settle(model, (8.0 + 0.5 * rng.standard_normal(5))[:, None], h=0.01, t=300.0)
    h, nsteps, q = 0.01, 2000, 5
    rows = []
    for m in [16, 32, 64, 128, 256]:
        N = 5 * m
        fns = {
            "SBR": lambda: sbr_lyapunov(model, x, m, h, nsteps, q, batch=16),
            "BQR": lambda: bqr_lyapunov(model, x, m, h, nsteps, q),
            "CQR": lambda: cqr_lyapunov(model, x, m, h, nsteps),
            "IRR": lambda: irr_lyapunov(model, x, m, h, nsteps, q),
            "SBD": lambda: sbd_lyapunov(model, x, m, h, nsteps, q),
        }
        for name, fn in fns.items():
            if N > LYAP_LIMITS[name]:
                continue
            L, t, mem = measure(fn, repeats=2 if name == "SBR" else 1)
            L = np.sort(np.concatenate([np.ravel(a) for a in (L if isinstance(L, list) else [L])]))[::-1]
            rows.append({"method": name, "N": N, "m": m, "time": t, "memory": mem,
                         "top": float(L[0]), "sum": float(np.sum(L)), "n": int(L.size)})
            print(f"lyap N={N:5d} {name:5s} t={t:9.4f}s mem={mem / 2 ** 20:10.3f}MB top={L[0]:.5f} sum={np.sum(L):.5f} n={L.size}",
                  flush=True)
            save_json("bench_lyap.json", rows)
    return rows


def orbit_bench():
    data = load_json("waves_F2.json")
    w = [v for v in data["waves"] if v["d"] == 9][0]
    model = Lorenz96(2.0)
    xs = np.array(w["x"])[:, None]
    d, tau, kap = 9, w["tau"], w["kappa"]
    rng = np.random.default_rng(5)
    rows = []
    for m in [4, 8, 16, 32]:
        N = d * m
        x0 = xs + 1e-3 * rng.standard_normal(xs.shape)
        _, t1, m1 = measure(lambda: reduced_shoot(model, x0, tau * 1.001, kap, 100), repeats=1)
        X0 = lift(x0, m)
        _, t2, m2 = measure(lambda: periodic_shoot(model, X0, d * tau * 1.001, d * 100), repeats=1)
        rows.append({"N": N, "reduced_time": t1, "reduced_mem": m1, "full_time": t2, "full_mem": m2})
        print(f"orbit N={N} reduced {t1:.3f}s {m1 / 2 ** 20:.3f}MB full {t2:.3f}s {m2 / 2 ** 20:.3f}MB", flush=True)
        save_json("bench_orbit.json", rows)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "wave"):
        wave_bench()
    if which in ("all", "lyap"):
        lyap_bench()
    if which in ("all", "orbit"):
        orbit_bench()
