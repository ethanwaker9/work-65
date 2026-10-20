from common import *
from sublattice.baselines import sbr_lyapunov, bqr_lyapunov, cqr_lyapunov
from sublattice.waves import settle


def main():
    model = Lorenz96(8.0)
    rng = np.random.default_rng(3)
    x = settle(model, (8.0 + 0.5 * rng.standard_normal(5))[:, None], h=0.01, t=300.0)
    out = {}
    for m in [4]:
        a = np.sort(np.concatenate(sbr_lyapunov(model, x, m, 0.01, 200000, 5)))[::-1]
        b = bqr_lyapunov(model, x, m, 0.01, 200000, 5)
        out[str(5 * m)] = {"sbr": a.tolist(), "bqr": b.tolist(), "maxdiff": float(np.max(np.abs(a - b))),
                           "sum_sbr": float(a.sum()), "sum_bqr": float(b.sum())}
        print(5 * m, np.round(a[:8], 4), np.round(b[:8], 4), "maxdiff", np.max(np.abs(a - b)), flush=True)
    save_json("lyap_accuracy.json", out)


if __name__ == "__main__":
    main()
