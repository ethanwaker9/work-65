from common import *
from sublattice.models import RosslerRing, LorenzRing, HindmarshRoseRing
from sublattice.waves import settle

ROWS = [
    {"name": "Lorenz", "cls": LorenzRing, "eps": 3.0, "x0": [1.0, 1.0, 20.0], "h": 0.004, "t": 60.0,
     "tr": 200.0, "sizes": [1, 2, 3, 4, 8, 16]},
    {"name": "Rossler", "cls": RosslerRing, "eps": 1.0, "x0": [1.0, 1.0, 0.5], "h": 0.02, "t": 300.0,
     "tr": 1500.0, "sizes": [1, 4, 12, 16, 20, 40]},
    {"name": "HindmarshRose", "cls": HindmarshRoseRing, "eps": 1.0, "x0": [-1.0, -5.0, 3.0], "h": 0.02,
     "t": 900.0, "tr": 3000.0, "sizes": [1, 3, 5, 6, 8, 16]},
]


def ring_panel(row, N, rng):
    model = row["cls"](row["eps"])
    base = settle(model, np.array([row["x0"]]), h=row["h"], t=600.0)
    x = np.tile(base, (N, 1)) + (1e-6 * rng.standard_normal((N, 3)) if N > 1 else 0.0)
    x = settle(model, x, h=row["h"], t=row["tr"])
    n = int(round(row["t"] / row["h"]))
    traj = np.zeros((n, 3))
    sync = np.zeros(n)
    for i in range(n):
        x = rk4_state(model, x, row["h"], 1)
        traj[i] = x[0]
        sync[i] = np.max(np.abs(x - x[0][None, :]))
    return traj[::2], float(np.median(sync))


def l96_panel(d, F, N, rng, delay=0.3, t=120.0, tr=2500.0):
    model = Lorenz96(F)
    base = settle(model, (F + 0.3 * np.random.default_rng(5).standard_normal(d))[:, None], h=0.01, t=800.0)
    x = np.tile(base, (N // d, 1)) + (1e-6 * rng.standard_normal((N, 1)) if N > d else 0.0)
    x = settle(model, x, h=0.01, t=tr)
    lag = int(round(delay / 0.01))
    n = int(round(t / 0.01)) + 2 * lag
    s = np.zeros(n)
    dev = np.zeros(n)
    for i in range(n):
        x = rk4_state(model, x, 0.01, 1)
        s[i] = x[0, 0]
        dev[i] = np.max(np.abs(np.roll(x, -d, axis=0) - x))
    traj = np.stack([s[2 * lag:], s[lag:-lag], s[:-2 * lag]], axis=1)
    return traj[::2], float(np.median(dev))


def main():
    rng = np.random.default_rng(42)
    out = {}
    for row in ROWS:
        panels = []
        for N in row["sizes"]:
            traj, sync = ring_panel(row, N, rng)
            panels.append({"N": N, "traj": traj, "sync": sync})
            print(row["name"], N, "sync error", sync, flush=True)
        out[row["name"]] = {"eps": row["eps"], "panels": panels}
        save_json("grid.json", out)
    panels = []
    for N in [5, 10, 15, 20, 30, 55]:
        traj, dev = l96_panel(5, 4.0, N, rng)
        panels.append({"N": N, "traj": traj, "sync": dev})
        print("L96", N, "deviation", dev, flush=True)
    out["Lorenz96"] = {"F": 4.0, "d": 5, "panels": panels}
    save_json("grid.json", out)


if __name__ == "__main__":
    main()
