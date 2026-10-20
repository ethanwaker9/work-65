from common import *
from sublattice.phasediff import phase_diffusion, phase_diffusion_green


def main():
    data = load_json("waves_F2.json")
    model = Lorenz96(data["F"])
    K = data["K"]
    rows = []
    for w in genuine_waves(data["waves"]):
        if w["d"] > 17:
            continue
        xs = np.array(w["x"])[:, None]
        n = max(4000, 200 * w["d"])
        a = phase_diffusion(model, xs, w["T"], n=n)
        b = phase_diffusion_green(model, xs, w["T"], n=n)
        fd = []
        for th in [0.02, 0.01, 0.005]:
            mu = wave_bloch_multipliers(model, xs, w["tau"], w["kappa"], 2 * K, np.array([th]))[0]
            top = mu[np.argmin(np.abs(mu - 1.0))]
            fd.append(float(-np.log(np.abs(top)) / (w["T"] * th * th)))
            cg = float(np.angle(top * np.exp(-1j * 0.0)) / (w["T"] * th))
        rows.append({"d": w["d"], "l": w["l"], "T": w["T"], "D_dyson": a["D"], "D_green": b["D"],
                     "D_fd": fd, "c": a["c"], "c_fd": cg})
        print(rows[-1], flush=True)
    save_json("phasediff.json", rows)


if __name__ == "__main__":
    main()
