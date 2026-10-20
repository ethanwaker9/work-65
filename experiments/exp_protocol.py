from common import *


def classify(x, tol):
    return minimal_spatial_period(x, tol)


def main():
    model = Lorenz96(2.0)
    N = 144
    out = {}
    for amp in [0.01, 0.1, 1.0]:
        rng = np.random.default_rng(int(1000 * amp) + 7)
        X = 2.0 + amp * rng.standard_normal((200, N, 1))
        X = rk4_state(model, X, 0.01, 300000)
        per = [classify(X[i], 1e-3) for i in range(X.shape[0])]
        v, c = np.unique(per, return_counts=True)
        out[str(amp)] = {int(a): int(b) for a, b in zip(v, c)}
        print(amp, out[str(amp)], flush=True)
    save_json("protocol144.json", out)


if __name__ == "__main__":
    main()
