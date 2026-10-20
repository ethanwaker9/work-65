from common import *
from sublattice.waves import settle
from sublattice.bloch import bloch_lyapunov


def main():
    data = load_json("route_d5.json")
    rows = data["rows"]
    for row in rows:
        if row["kind"] == "periodic" and row["base_exps"][0] > 1e-3:
            F = row["F"]
            model = Lorenz96(F)
            rng = np.random.default_rng(int(F * 100))
            x = settle(model, (F + 0.3 * rng.standard_normal(5))[:, None], h=0.01, t=600.0)
            fr = [tuple(f) for f in row["farey"]]
            th = np.concatenate([np.array([2 * np.pi * a / b for a, b in fr]), np.array(row["grid"])])
            L, _ = bloch_lyapunov(model, x, th, 0.01, 30000, 5)
            lam = L[:, 0]
            row["kind"] = "chaotic"
            row["T"] = None
            row["base_exps"] = L[len(fr)].tolist()
            row["lam_farey"] = lam[:len(fr)].tolist()
            row["lam_grid"] = lam[len(fr):].tolist()
            row["stable"] = {str(m): bool(all(lam[i] < 0 for i, (a, b) in enumerate(fr) if m % b == 0)) for m in range(2, 21)}
            print("fixed", F, np.round(row["base_exps"][:3], 4), flush=True)
    save_json("route_d5.json", data)
    st = {}
    for m in range(2, 21):
        lost = None
        for row in rows:
            if row["F"] < 3.0:
                continue
            if not row["stable"][str(m)]:
                lost = row["F"]
                break
        st[m] = lost
    print("first loss above F=3:", st, flush=True)
    early = [(row["F"], [m for m in range(2, 21) if not row["stable"][str(m)]]) for row in rows if row["F"] < 1.2]
    print(early, flush=True)


if __name__ == "__main__":
    main()
