from common import *
from sublattice.waves import reduced_shoot


def continue_wave(model_F, w, Fs, K):
    xs = np.array(w["x"])[:, None]
    tau = w["tau"]
    kap = w["kappa"]
    out = {}
    for F in Fs:
        model = Lorenz96(F)
        xs, tau, res = reduced_shoot(model, xs, tau, kap, K)
        if not np.isfinite(res) or res > 1e-9:
            break
        out[float(F)] = (xs.copy(), tau)
    return out


def main():
    data = load_json("waves_F2.json")
    w = [v for v in data["waves"] if v["d"] == 9][0]
    K = 200
    up = np.round(np.arange(2.0, 3.6001, 0.05), 4)
    down = np.round(np.arange(2.0, 0.9499, -0.05), 4)
    sols = continue_wave(2.0, w, up, K)
    sols.update(continue_wave(2.0, w, down, K))
    Fs = sorted(sols)
    thetas = np.linspace(0.0, np.pi, 61)
    eta = []
    for F in Fs:
        xs, tau = sols[F]
        mus = wave_bloch_multipliers(Lorenz96(F), xs, tau, w["kappa"], K, thetas)
        T = 9 * tau
        row = [np.log(nontrivial_radius(mu)) / T if i == 0 else np.log(np.max(np.abs(mu))) / T for i, mu in enumerate(mus)]
        eta.append(row)
        print(F, round(T, 4), round(max(row[1:]), 5), flush=True)
    save_json("surface.json", {"d": 9, "kappa": w["kappa"], "F": Fs, "theta": thetas, "eta": eta})


if __name__ == "__main__":
    main()
