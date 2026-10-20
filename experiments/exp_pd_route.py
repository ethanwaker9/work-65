from common import *
from sublattice.waves import settle, periodic_shoot
from sublattice.bloch import orbit_bloch_multipliers
from sublattice.phasediff import phase_diffusion


def main():
    rng = np.random.default_rng(1)
    F0 = 3.97
    model = Lorenz96(F0)
    x = settle(model, (F0 + 0.3 * rng.standard_normal(5))[:, None], h=0.01, t=3000.0)
    xs, T, res = periodic_shoot(model, x, 4.73, 2400)
    out = []
    Fs = list(np.round(np.arange(3.97, 4.0601, 0.01), 4)) + list(np.round(np.arange(3.96, 3.9379, -0.004), 4))
    state = {3.97: (xs, T)}
    for F in Fs:
        model = Lorenz96(F)
        near = min(state, key=lambda f: abs(f - F))
        x0, T0 = state[near]
        xs, T, res = periodic_shoot(model, x0, T0, 2400)
        if res > 1e-9:
            print(F, "failed", res, flush=True)
            continue
        state[F] = (xs, T)
        mu0 = orbit_bloch_multipliers(model, xs, T, 2400, np.array([0.0]))[0]
        mod = np.sort(np.abs(mu0))[::-1]
        pdv = phase_diffusion(model, xs, T, n=4800)
        out.append({"F": float(F), "T": T, "mu2": float(mod[1]), "D": pdv["D"], "c": pdv["c"]})
        print(F, round(T, 5), "mu2", round(mod[1], 5), "D", pdv["D"], flush=True)
    out.sort(key=lambda r: r["F"])
    save_json("pd_route.json", out)


if __name__ == "__main__":
    main()
