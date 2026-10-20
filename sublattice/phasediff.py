import numpy as np
from .bloch import Stencil, rk4_real_variational


def moment_matrices(model, x, st):
    J = model.jac_blocks(x)
    d, k = x.shape
    B1 = np.zeros((d * k, d * k))
    B2 = np.zeros((d * k, d * k))
    for a in range(st.offsets.size):
        for j in range(d):
            q = st.q[a, j]
            if q != 0:
                jj = st.jj[a, j]
                B1[j * k:(j + 1) * k, jj * k:(jj + 1) * k] += q * J[a, j]
                B2[j * k:(j + 1) * k, jj * k:(jj + 1) * k] += q * q * J[a, j]
    return B1, B2


def wave_trajectory(model, x0, T, n):
    d, k = x0.shape
    st = Stencil(model.offsets, d)
    xs = np.zeros((n + 1, d, k))
    Phis = np.zeros((n + 1, d * k, d * k))
    xs[0] = x0
    Phis[0] = np.eye(d * k)
    x = x0.copy()
    U = np.eye(d * k).reshape(d, k, d * k)
    h = T / n
    for i in range(n):
        x, U = rk4_real_variational(model, x, U, h, 1, st)
        xs[i + 1] = x
        Phis[i + 1] = U.reshape(d * k, d * k)
    return xs, Phis, st


def simpson_weights(n, h):
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0


def cumulative_trapezoid(Y, h):
    out = np.zeros_like(Y)
    out[1:] = np.cumsum(0.5 * h * (Y[1:] + Y[:-1]), axis=0)
    return out


def _floquet_data(model, x0, T, n):
    if n % 2:
        n += 1
    xs, Phis, st = wave_trajectory(model, x0, T, n)
    dk = Phis.shape[1]
    M0 = Phis[-1]
    v = model.rhs(xs[0]).ravel()
    ev, W = np.linalg.eig(M0.T)
    i = np.argmin(np.abs(ev - 1.0))
    w = np.real(W[:, i])
    w = w / (w @ v)
    Pi = np.outer(v, w)
    Z = np.linalg.inv(np.eye(dk) - M0 + Pi) - Pi
    Pinv = np.linalg.inv(Phis)
    B = [moment_matrices(model, xs[t], st) for t in range(n + 1)]
    B1 = np.array([b[0] for b in B])
    B2 = np.array([b[1] for b in B])
    return n, T / n, xs, Phis, Pinv, M0, v, w, Z, B1, B2


def phase_diffusion(model, x0, T, n=4000):
    n, h, xs, Phis, Pinv, M0, v, w, Z, B1, B2 = _floquet_data(model, x0, T, n)
    C1 = Pinv @ B1 @ Phis
    C2 = Pinv @ B2 @ Phis
    wts = simpson_weights(n, h)
    I1 = np.tensordot(wts, C1, axes=1)
    I2 = np.tensordot(wts, C2, axes=1)
    Kc = cumulative_trapezoid(C1, h)
    I11 = np.tensordot(wts, C1 @ Kc, axes=1)
    M1 = M0 @ I1
    M2 = M0 @ I11 + 0.5 * M0 @ I2
    a1 = w @ M1 @ v
    a2 = w @ M2 @ v + w @ M1 @ Z @ M1 @ v
    return {"alpha1": float(a1), "alpha2": float(a2), "T": float(T),
            "c": float(a1 / T), "D": float((a2 - 0.5 * a1 * a1) / T)}


def phase_diffusion_green(model, x0, T, n=4000):
    n, h, xs, Phis, Pinv, M0, v, w, Z, B1, B2 = _floquet_data(model, x0, T, n)
    xdot = np.array([model.rhs(xs[t]).ravel() for t in range(n + 1)])
    psi = np.einsum('i,tij->tj', w, Pinv)
    beta = np.einsum('ti,tij,tj->t', psi, B1, xdot)
    gvec = np.einsum('tij,tj->ti', B1, xdot) - xdot * beta[:, None]
    Y = np.einsum('tij,tj->ti', Pinv, gvec)
    wts = simpson_weights(n, h)
    G = wts @ Y
    tail = M0 @ Z @ G
    inner = np.einsum('tij,tj->ti', Phis, cumulative_trapezoid(Y, h) + tail[None, :])
    total = wts @ np.einsum('ti,tij,tj->t', psi, B1, inner)
    term2 = wts @ np.einsum('ti,tij,tj->t', psi, B2, xdot)
    return {"D": float((0.5 * term2 + total) / T), "c": float((wts @ beta) / T), "G_dot_w": float(w @ G)}
