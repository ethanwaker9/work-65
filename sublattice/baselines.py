import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigs
from .bloch import (Stencil, real_apply, rk4_state, rk4_real_variational, rk4_bloch,
                    identity_frame, shift_matrix, wave_bloch_multipliers,
                    lattice_wave_multipliers)


def lift(x, m):
    return np.tile(x, (m, 1))


def sbr_batch(m):
    return 4


def sbr_wave(model, xd, tau, kappa, m, K, batch=None):
    b = sbr_batch(m) if batch is None else batch
    return lattice_wave_multipliers(model, xd, tau, kappa, K, m, batch=b)


def fm_wave(model, xd, T, m, steps):
    x = lift(xd, m)
    N, k = x.shape
    st = Stencil(model.offsets, N)
    U = np.eye(N * k).reshape(N, k, N * k)
    _, U = rk4_real_variational(model, x, U, T / steps, steps, st)
    return np.linalg.eigvals(U.reshape(N * k, N * k))


def rm_wave(model, xd, tau, kappa, m, K):
    d = xd.shape[0]
    x = lift(xd, m)
    N, k = x.shape
    st = Stencil(model.offsets, N)
    U = np.eye(N * k).reshape(N, k, N * k)
    _, U = rk4_real_variational(model, x, U, tau / K, K, st)
    R = np.roll(U, kappa, axis=0).reshape(N * k, N * k)
    Rd = np.linalg.matrix_power(R, d)
    M = np.roll(Rd.reshape(N, k, N * k), -d * kappa, axis=0).reshape(N * k, N * k)
    return np.linalg.eigvals(M)


def structure_list(offsets, d, N):
    out = []
    for a, s in enumerate(offsets):
        for j in range(d):
            rows = np.arange(j, N, d)
            cols = (rows + s) % N
            out.append((a, j, rows, cols))
    return out


def irr_basis(N, d):
    m = N // d
    I = np.eye(N)
    T = np.zeros((N, N), dtype=complex)
    for r in range(m):
        th = 2.0 * np.pi * r / m
        Pr = np.zeros((N, N), dtype=complex)
        for kk in range(m):
            Pr += np.exp(-1j * th * kk) * np.roll(I, kk * d, axis=1)
        Pr /= m
        B = Pr[:, :d]
        Q, _ = np.linalg.qr(B)
        T[:, r * d:(r + 1) * d] = Q
    return T


def sbd_basis(mats, N, rng=None, tol=1e-9):
    rng = np.random.default_rng(0) if rng is None else rng
    Bh = csr_matrix((N, N), dtype=complex)
    for Bk in mats:
        c, e = rng.standard_normal(2)
        Bh = Bh + c * (Bk + Bk.conj().T) + 1j * e * (Bk - Bk.conj().T)
    _, V = np.linalg.eigh(Bh.toarray())
    P = np.zeros((N, 0), dtype=complex)
    blocks = []
    adj = [Bk.conj().T.tocsr() for Bk in mats]
    while P.shape[1] < N:
        res = V - P @ (P.conj().T @ V)
        j = int(np.argmax(np.linalg.norm(res, axis=0)))
        v = res[:, j] / np.linalg.norm(res[:, j])
        W = v[:, None]
        cand = [v] + [Bk @ v for Bk in mats] + [Ak @ v for Ak in adj]
        W = _extend(W, cand[1:], P, tol)
        while True:
            u = W @ (rng.standard_normal(W.shape[1]) + 1j * rng.standard_normal(W.shape[1]))
            imgs = [Bk @ u for Bk in mats] + [Ak @ u for Ak in adj]
            W2 = _extend(W, imgs, P, tol)
            if W2.shape[1] == W.shape[1]:
                break
            W = W2
        blocks.append(W.shape[1])
        P = np.hstack([P, W])
    return P, blocks


def _extend(W, vecs, P, tol):
    for u in vecs:
        u = u - P @ (P.conj().T @ u)
        u = u - W @ (W.conj().T @ u)
        u = u - W @ (W.conj().T @ u)
        nu = np.linalg.norm(u)
        if nu > tol:
            W = np.hstack([W, (u / nu)[:, None]])
    return W


def transformed_structure(T, sizes, struct, N):
    blocks = []
    start = 0
    for sz in sizes:
        Tb = T[:, start:start + sz]
        Cs = []
        for (a, j, rows, cols) in struct:
            Cs.append(Tb[rows].conj().T @ Tb[cols])
        blocks.append(np.array(Cs))
        start += sz
    return blocks


def _assemble(Cb, Jflat, k):
    n = Cb.shape[1]
    A = np.einsum('sab,spq->apbq', Cb, Jflat).reshape(n * k, n * k)
    return A


def _jflat(model, x, d):
    J = model.jac_blocks(x)
    return J.reshape(-1, J.shape[-2], J.shape[-1])


def block_propagate(model, xd, Cblocks, T, steps, k):
    d = xd.shape[0]
    h = T / steps
    out = []
    for Cb in Cblocks:
        n = Cb.shape[1]
        x = xd.copy()
        U = np.eye(n * k, dtype=complex)
        for _ in range(steps):
            k1 = model.rhs(x)
            A1 = _assemble(Cb, _jflat(model, x, d), k)
            x2 = x + 0.5 * h * k1
            A2 = _assemble(Cb, _jflat(model, x2, d), k)
            k2 = model.rhs(x2)
            x3 = x + 0.5 * h * k2
            A3 = _assemble(Cb, _jflat(model, x3, d), k)
            k3 = model.rhs(x3)
            x4 = x + h * k3
            A4 = _assemble(Cb, _jflat(model, x4, d), k)
            k4 = model.rhs(x4)
            L1 = A1 @ U
            L2 = A2 @ (U + 0.5 * h * L1)
            L3 = A3 @ (U + 0.5 * h * L2)
            L4 = A4 @ (U + h * L3)
            U = U + (h / 6.0) * (L1 + 2.0 * L2 + 2.0 * L3 + L4)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(np.linalg.eigvals(U))
    return np.concatenate(out)


def irr_wave(model, xd, T, m, steps):
    d, k = xd.shape
    N = d * m
    Tm = irr_basis(N, d)
    struct = structure_list(model.offsets, d, N)
    Cb = transformed_structure(Tm, [d] * m, struct, N)
    return block_propagate(model, xd, Cb, T, steps, k)


def structure_matrices(offsets, d, N):
    mats = []
    for (a, j, rows, cols) in structure_list(offsets, d, N):
        mats.append(csr_matrix((np.ones(rows.size, dtype=complex), (rows, cols)), shape=(N, N)))
    return mats


def sbd_wave(model, xd, T, m, steps):
    d, k = xd.shape
    N = d * m
    mats = structure_matrices(model.offsets, d, N)
    P, sizes = sbd_basis(mats, N)
    struct = structure_list(model.offsets, d, N)
    Cb = transformed_structure(P, sizes, struct, N)
    return block_propagate(model, xd, Cb, T, steps, k)


def nk_wave(model, xd, T, m, steps, nev=6, tol=1e-10):
    x0 = lift(xd, m)
    N, k = x0.shape
    st = Stencil(model.offsets, N)
    h = T / steps

    def mv(v):
        U = np.asarray(v, dtype=float).reshape(N, k, 1)
        _, U = rk4_real_variational(model, x0.copy(), U, h, steps, st)
        return U.ravel()

    op = LinearOperator((N * k, N * k), matvec=mv, dtype=float)
    vals = eigs(op, k=nev, which='LM', tol=tol, return_eigenvectors=False)
    return vals


def hill_wave(model, xd, T, m, nharm=16, nsamp=None):
    d, k = xd.shape
    N = d * m
    nsamp = 8 * nharm if nsamp is None else nsamp
    steps_per = 8
    x = xd.copy()
    Js = []
    for i in range(nsamp):
        Js.append(model.jac_blocks(x))
        x = rk4_state(model, x, T / (nsamp * steps_per), steps_per)
    Js = np.array(Js)
    Jh = np.fft.fft(Js, axis=0) / nsamp
    st = Stencil(model.offsets, N)
    nK = 2 * nharm + 1
    om = 2.0 * np.pi / T
    H = np.zeros((nK * N * k, nK * N * k), dtype=complex)
    sites = np.arange(N)
    for p in range(nK):
        for q in range(nK):
            n = (p - nharm) - (q - nharm)
            if abs(n) > 2 * nharm:
                continue
            Jn = Jh[n % nsamp]
            r0 = p * N * k
            c0 = q * N * k
            for a in range(st.offsets.size):
                blk = Jn[a][sites % d]
                for i_ in range(k):
                    for j_ in range(k):
                        H[r0 + sites * k + i_, c0 + st.jj[a] * k + j_] += blk[:, i_, j_]
        H[p * N * k:(p + 1) * N * k, p * N * k:(p + 1) * N * k] -= 1j * (p - nharm) * om * np.eye(N * k)
    lam = np.linalg.eigvals(H)
    idx = np.argsort(np.abs(lam.imag))[:N * k]
    return np.exp(lam[idx] * T)


def sbr_lyapunov(model, xd, m, h, nsteps, qr_every=5, batch=None):
    from .bloch import bloch_lyapunov
    r = np.arange(m // 2 + 1)
    thetas = 2.0 * np.pi * r / m
    b = sbr_batch(m) if batch is None else batch
    out = []
    for b0 in range(0, thetas.size, b):
        L, _ = bloch_lyapunov(model, xd, thetas[b0:b0 + b], h, nsteps, qr_every)
        out.append(L)
    half = np.vstack(out)
    full = [half[i] for i in range(half.shape[0])]
    for rr in range(m // 2 + 1, m):
        full.append(half[m - rr])
    return np.array(full)


def bqr_lyapunov(model, xd, m, h, nsteps, qr_every=5):
    x = lift(xd, m)
    N, k = x.shape
    st = Stencil(model.offsets, N)
    U = np.eye(N * k).reshape(N, k, N * k)
    S = np.zeros(N * k)
    nb = nsteps // qr_every
    for _ in range(nb):
        x, U = rk4_real_variational(model, x, U, h, qr_every, st)
        Q, R = np.linalg.qr(U.reshape(N * k, N * k))
        S += np.log(np.abs(np.diag(R)))
        U = Q.reshape(N, k, N * k)
    return np.sort(S / (nb * qr_every * h))[::-1]


def _skew_part(M):
    L = np.tril(M, -1)
    return L - L.T


def cqr_lyapunov(model, xd, m, h, nsteps):
    x = lift(xd, m)
    N, k = x.shape
    st = Stencil(model.offsets, N)
    Q = np.eye(N * k)
    S = np.zeros(N * k)

    def F(xx, QQ):
        J = model.jac_blocks(xx)
        AQ = real_apply(J, QQ.reshape(N, k, N * k), st).reshape(N * k, N * k)
        B = QQ.T @ AQ
        return QQ @ _skew_part(B), np.diag(B).copy()

    for _ in range(nsteps):
        k1 = model.rhs(x)
        F1, d1 = F(x, Q)
        x2 = x + 0.5 * h * k1
        F2, d2 = F(x2, Q + 0.5 * h * F1)
        k2 = model.rhs(x2)
        x3 = x + 0.5 * h * k2
        F3, d3 = F(x3, Q + 0.5 * h * F2)
        k3 = model.rhs(x3)
        x4 = x + h * k3
        F4, d4 = F(x4, Q + h * F3)
        k4 = model.rhs(x4)
        Q = Q + (h / 6.0) * (F1 + 2.0 * F2 + 2.0 * F3 + F4)
        S += (h / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        Q, R = np.linalg.qr(Q)
        sg = np.sign(np.diag(R))
        Q = Q * sg[None, :]
    return np.sort(S / (nsteps * h))[::-1]


def block_lyapunov(model, xd, Cblocks, h, nsteps, qr_every, k):
    d = xd.shape[0]
    out = []
    for Cb in Cblocks:
        n = Cb.shape[1]
        x = xd.copy()
        U = np.eye(n * k, dtype=complex)
        S = np.zeros(n * k)
        nb = nsteps // qr_every
        for _ in range(nb):
            for _s in range(qr_every):
                k1 = model.rhs(x)
                A1 = _assemble(Cb, _jflat(model, x, d), k)
                x2 = x + 0.5 * h * k1
                A2 = _assemble(Cb, _jflat(model, x2, d), k)
                k2 = model.rhs(x2)
                x3 = x + 0.5 * h * k2
                A3 = _assemble(Cb, _jflat(model, x3, d), k)
                k3 = model.rhs(x3)
                x4 = x + h * k3
                A4 = _assemble(Cb, _jflat(model, x4, d), k)
                k4 = model.rhs(x4)
                L1 = A1 @ U
                L2 = A2 @ (U + 0.5 * h * L1)
                L3 = A3 @ (U + 0.5 * h * L2)
                L4 = A4 @ (U + h * L3)
                U = U + (h / 6.0) * (L1 + 2.0 * L2 + 2.0 * L3 + L4)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Q, R = np.linalg.qr(U)
            S += np.log(np.abs(np.diag(R)))
            U = Q
        out.append(np.sort(S / (nb * qr_every * h))[::-1])
    return out


def irr_lyapunov(model, xd, m, h, nsteps, qr_every=5):
    d, k = xd.shape
    N = d * m
    Tm = irr_basis(N, d)
    struct = structure_list(model.offsets, d, N)
    Cb = transformed_structure(Tm, [d] * m, struct, N)
    return np.array(block_lyapunov(model, xd, Cb, h, nsteps, qr_every, k))


def sbd_lyapunov(model, xd, m, h, nsteps, qr_every=5):
    d, k = xd.shape
    N = d * m
    mats = structure_matrices(model.offsets, d, N)
    P, sizes = sbd_basis(mats, N)
    struct = structure_list(model.offsets, d, N)
    Cb = transformed_structure(P, sizes, struct, N)
    return block_lyapunov(model, xd, Cb, h, nsteps, qr_every, k)
