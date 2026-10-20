from common import *
from sublattice.waves import settle, detect_period, periodic_shoot
from sublattice.bloch import orbit_bloch_multipliers, bloch_lyapunov
from sublattice.admissible import farey_phases

D = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MMAX = 20


def main():
    fr = farey_phases(MMAX)
    th_f = np.array([2.0 * np.pi * a / b for a, b in fr])
    grid = np.linspace(0.0, np.pi, 73)
    thetas = np.concatenate([th_f, grid])
    Fs = np.round(np.arange(0.90, 7.0001, 0.02), 4)
    rng = np.random.default_rng(7)
    x = (Fs[0] + 0.3 * rng.standard_normal(D))[:, None]
    rows = []
    for F in Fs:
        model = Lorenz96(F)
        x = settle(model, x, h=0.01, t=400.0)
        det = detect_period(model, x, h=0.002, tmax=100.0, tol=1e-2)
        kind = "chaotic"
        if det is not None:
            xs, Ts = det
            kind = "periodic"
            x = xs
            n = max(400, int(np.ceil(Ts / 0.004)))
            mus = orbit_bloch_multipliers(model, xs, Ts, n, thetas, batch=160)
            lam = np.array([np.log(nontrivial_radius(mu)) / Ts if abs(t) < 1e-14 else np.log(np.max(np.abs(mu))) / Ts
                            for mu, t in zip(mus, thetas)])
            base_exps = np.sort(np.log(np.abs(mus[np.argmin(np.abs(thetas))])) / Ts)[::-1]
        if kind == "chaotic":
            L, _ = bloch_lyapunov(model, x, thetas, 0.01, 30000, qr_every=5)
            lam = L[:, 0]
            base_exps = L[np.argmin(np.abs(thetas))]
            Ts = None
        lam_f = lam[:len(fr)]
        stable = {}
        for m in range(2, MMAX + 1):
            ok = all(lam_f[i] < 0 for i, (a, b) in enumerate(fr) if m % b == 0)
            stable[m] = bool(ok)
        rows.append({"F": float(F), "kind": kind, "T": Ts, "base_exps": base_exps, "grid": grid,
                     "lam_grid": lam[len(fr):], "farey": fr, "lam_farey": lam_f, "stable": stable})
        print(f"F={F:.2f} {kind} T={Ts} base={np.round(base_exps[:3], 4)} unstable_m={[m for m in stable if not stable[m]][:10]}",
              flush=True)
        save_json(f"route_d{D}.json", {"d": D, "rows": rows})
    save_json(f"route_d{D}.json", {"d": D, "rows": rows})


if __name__ == "__main__":
    main()
