from common import *
from sublattice.phasediff import phase_diffusion
from sublattice.admissible import bad_denominators, admissible_multiples, minimal_bad

F = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
DMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 48
DMIN = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SEEDS = 12
K = 200
MFAREY = 64


def main():
    model = Lorenz96(F)
    out = []
    fname = f"waves_F{F:g}.json"
    if DMIN > 4 and os.path.exists(os.path.join(RESULTS, fname)):
        out = [w for w in load_json(fname)["waves"] if w["d"] < DMIN]
    grid = np.linspace(0.0, np.pi, 361)
    for d in range(DMIN, DMAX + 1):
        t0 = time.time()
        waves = find_waves(model, d, SEEDS, K=K)
        for w in waves:
            xs = w["x"][:, None]
            mus = wave_bloch_multipliers(model, xs, w["tau"], w["kappa"], K, grid)
            sym = [nontrivial_radius(mu) if i == 0 else float(np.max(np.abs(mu))) for i, mu in enumerate(mus)]
            pd = phase_diffusion(model, xs, w["T"], n=max(2000, 40 * d))

            def symbol_many(th):
                mu = wave_bloch_multipliers(model, xs, w["tau"], w["kappa"], K, th, batch=128)
                return np.max(np.abs(mu), axis=1)

            bad, vals = bad_denominators(symbol_many, MFAREY)
            adm = admissible_multiples(bad, MFAREY)
            w.update({"grid": grid, "symbol": sym, "D": pd["D"], "c": pd["c"],
                      "bad": sorted(bad), "minimal_bad": minimal_bad(bad), "admissible": adm,
                      "universal_on_grid": bool(max(sym[1:]) < 1.0) and pd["D"] > 0,
                      "farey_max": max(vals.values())})
            out.append(w)
            print(f"d={d} kappa={w['kappa']} l={w['l']} T={w['T']:.5f} rho0={w['rho0']:.4f} "
                  f"maxsym={max(sym[1:]):.5f} D={pd['D']:.5f} minimal_bad={w['minimal_bad'][:6]} "
                  f"adm<=16={[a for a in adm if a <= 16]}", flush=True)
        print(f"d={d} done in {time.time() - t0:.1f}s, {len(waves)} waves", flush=True)
        save_json(fname, {"F": F, "K": K, "Mfarey": MFAREY, "waves": out})


if __name__ == "__main__":
    main()
