import numpy as np


class Stencil:
    def __init__(self, offsets, d):
        self.offsets = np.asarray(offsets, dtype=int)
        self.d = int(d)
        idx = np.arange(self.d)[None, :] + self.offsets[:, None]
        self.q = np.floor_divide(idx, self.d)
        self.jj = idx - self.q * self.d

    def phases(self, thetas):
        thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
        return np.exp(1j * thetas[:, None, None] * self.q[None, :, :])


def twisted_apply(J, U, st, ph):
    if J.shape[-1] == 1:
        coef = ph * J[None, :, :, 0, 0]
        return np.einsum('Ksd,Ksdc->Kdc', coef, U[:, st.jj, 0, :])[:, :, None, :]
    G = U[:, st.jj]
    return np.einsum('Ksd,sdpq,Ksdqc->Kdpc', ph, J, G)


def real_apply(J, U, st):
    if J.shape[-1] == 1:
        return np.einsum('sd,sdc->dc', J[:, :, 0, 0], U[st.jj, 0, :])[:, None, :]
    return np.einsum('sdpq,sdqc->dpc', J, U[st.jj])


def rk4_state(model, x, h, nsteps):
    for _ in range(nsteps):
        k1 = model.rhs(x)
        k2 = model.rhs(x + 0.5 * h * k1)
        k3 = model.rhs(x + 0.5 * h * k2)
        k4 = model.rhs(x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def rk4_bloch(model, x, U, h, nsteps, st, ph):
    h2 = 0.5 * h
    h6 = h / 6.0
    for _ in range(nsteps):
        k1, J1 = model.rhs_jac(x)
        L = twisted_apply(J1, U, st, ph)
        acc = L.copy()
        k2, J2 = model.rhs_jac(x + h2 * k1)
        L = twisted_apply(J2, U + h2 * L, st, ph)
        acc += 2.0 * L
        k3, J3 = model.rhs_jac(x + h2 * k2)
        L = twisted_apply(J3, U + h2 * L, st, ph)
        acc += 2.0 * L
        k4, J4 = model.rhs_jac(x + h * k3)
        L = twisted_apply(J4, U + h * L, st, ph)
        acc += L
        x = x + h6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        U = U + h6 * acc
    return x, U


def rk4_real_variational(model, x, U, h, nsteps, st):
    h2 = 0.5 * h
    h6 = h / 6.0
    for _ in range(nsteps):
        k1, J1 = model.rhs_jac(x)
        L = real_apply(J1, U, st)
        acc = L.copy()
        k2, J2 = model.rhs_jac(x + h2 * k1)
        L = real_apply(J2, U + h2 * L, st)
        acc += 2.0 * L
        k3, J3 = model.rhs_jac(x + h2 * k2)
        L = real_apply(J3, U + h2 * L, st)
        acc += 2.0 * L
        k4, J4 = model.rhs_jac(x + h * k3)
        L = real_apply(J4, U + h * L, st)
        acc += L
        x = x + h6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        U = U + h6 * acc
    return x, U


def identity_frame(d, k, K):
    E = np.zeros((K, d, k, d * k), dtype=complex)
    for j in range(d):
        for c in range(k):
            E[:, j, c, j * k + c] = 1.0
    return E


def shift_matrix(d, k, kappa, theta):
    P = np.zeros((d * k, d * k), dtype=complex)
    for j in range(d):
        i = j + kappa
        q = i // d
        jj = i - q * d
        P[j * k:(j + 1) * k, jj * k:(jj + 1) * k] = np.exp(1j * theta * q) * np.eye(k)
    return P


def bloch_fundamental(model, x0, tau, nsteps, thetas):
    d, k = x0.shape
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    st = Stencil(model.offsets, d)
    ph = st.phases(thetas)
    if np.all(np.abs(ph.imag) < 1e-15):
        ph = ph.real
        U = identity_frame(d, k, thetas.size).real.copy()
    else:
        U = identity_frame(d, k, thetas.size)
    x, U = rk4_bloch(model, x0.copy(), U, tau / nsteps, nsteps, st, ph)
    return x, U.reshape(thetas.size, d * k, d * k)


def half_phases(m):
    r = np.arange(m // 2 + 1)
    return r, 2.0 * np.pi * r / m


def wave_bloch_multipliers(model, x0, tau, kappa, nsteps, thetas, batch=64):
    d, k = x0.shape
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    out = np.zeros((thetas.size, d * k), dtype=complex)
    order = np.argsort(np.abs(np.sin(thetas)) >= 1e-14, kind="stable")
    for b0 in range(0, order.size, batch):
        idx = order[b0:b0 + batch]
        th = thetas[idx]
        _, Phi = bloch_fundamental(model, x0, tau, nsteps, th)
        for t, i in enumerate(idx):
            P = shift_matrix(d, k, kappa, th[t])
            nu = np.linalg.eigvals(P.conj().T @ Phi[t])
            out[i] = np.exp(1j * th[t] * kappa) * nu ** d
    return out


def lattice_wave_multipliers(model, x0, tau, kappa, nsteps, m, batch=64):
    r, th = half_phases(m)
    half = wave_bloch_multipliers(model, x0, tau, kappa, nsteps, th, batch=batch)
    full = np.zeros((m, half.shape[1]), dtype=complex)
    full[r] = half
    for rr in range(m // 2 + 1, m):
        full[rr] = np.conj(half[m - rr])
    return full


def orbit_bloch_multipliers(model, x0, T, nsteps, thetas, batch=64):
    d, k = x0.shape
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    out = np.zeros((thetas.size, d * k), dtype=complex)
    for b0 in range(0, thetas.size, batch):
        th = thetas[b0:b0 + batch]
        _, Phi = bloch_fundamental(model, x0, T, nsteps, th)
        out[b0:b0 + th.size] = np.linalg.eigvals(Phi)
    return out


def bloch_symbol(multipliers, theta, trivial_tol=1e-6):
    mods = np.abs(multipliers)
    if abs(np.angle(np.exp(1j * theta))) < 1e-14:
        i = np.argmin(np.abs(multipliers - 1.0))
        mods = np.delete(mods, i)
    return mods.max()


def bloch_lyapunov(model, x0, thetas, h, nsteps, qr_every=5, transient=0):
    d, k = x0.shape
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    st = Stencil(model.offsets, d)
    ph = st.phases(thetas)
    x = rk4_state(model, x0.copy(), h, transient)
    if np.all(np.abs(ph.imag) < 1e-15):
        ph = ph.real
        U = identity_frame(d, k, thetas.size).real.copy()
    else:
        U = identity_frame(d, k, thetas.size)
    S = np.zeros((thetas.size, d * k))
    nblocks = nsteps // qr_every
    for _ in range(nblocks):
        x, U = rk4_bloch(model, x, U, h, qr_every, st, ph)
        Q, R = np.linalg.qr(U.reshape(thetas.size, d * k, d * k))
        S += np.log(np.abs(np.diagonal(R, axis1=1, axis2=2)))
        U = Q.reshape(thetas.size, d, k, d * k)
    return np.sort(S / (nblocks * qr_every * h), axis=1)[:, ::-1], x
