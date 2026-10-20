import numpy as np


_INDEX_CACHE = {}


def shifted_indices(n, shifts):
    key = (n, tuple(shifts))
    if key not in _INDEX_CACHE:
        base = np.arange(n)
        _INDEX_CACHE[key] = [np.mod(base + s, n) for s in shifts]
    return _INDEX_CACHE[key]


class LatticeModel:
    name = "lattice"
    k = 1
    offsets = np.array([0])

    def rhs(self, x):
        raise NotImplementedError

    def jac_blocks(self, x):
        raise NotImplementedError

    def homogeneous_state(self):
        raise NotImplementedError

    def rhs_jac(self, x):
        return self.rhs(x), self.jac_blocks(x)


class Lorenz96(LatticeModel):
    name = "Lorenz-96"
    k = 1
    offsets = np.array([-2, -1, 0, 1])

    def __init__(self, F):
        self.F = float(F)

    def rhs(self, x):
        u = x[..., 0]
        im2, im1, ip1 = shifted_indices(u.shape[-1], (-2, -1, 1))
        v = (u[..., ip1] - u[..., im2]) * u[..., im1] - u + self.F
        return v[..., None]

    def jac_blocks(self, x):
        u = x[..., 0]
        im2, im1, ip1 = shifted_indices(u.shape[-1], (-2, -1, 1))
        um1 = u[..., im1]
        J = np.stack([-um1, u[..., ip1] - u[..., im2], -np.ones_like(u), um1], axis=-2)
        return J[..., None, None]

    def homogeneous_state(self):
        return np.array([self.F])

    def rhs_jac(self, x):
        u = x[..., 0]
        im2, im1, ip1 = shifted_indices(u.shape[-1], (-2, -1, 1))
        um1 = u[..., im1]
        a = u[..., ip1] - u[..., im2]
        v = a * um1 - u + self.F
        J = np.empty(u.shape[:-1] + (4, u.shape[-1], 1, 1))
        J[..., 0, :, 0, 0] = -um1
        J[..., 1, :, 0, 0] = a
        J[..., 2, :, 0, 0] = -1.0
        J[..., 3, :, 0, 0] = um1
        return v[..., None], J

    def dispersion(self, theta):
        return -1.0 + self.F * (np.exp(1j * theta) - np.exp(-2j * theta))


class DiffusiveRing(LatticeModel):
    offsets = np.array([-1, 0, 1])

    def __init__(self, eps, H):
        self.eps = float(eps)
        self.H = np.asarray(H, dtype=float)
        self.k = self.H.shape[0]

    def local(self, x):
        raise NotImplementedError

    def local_jac(self, x):
        raise NotImplementedError

    def rhs(self, x):
        im1, ip1 = shifted_indices(x.shape[-2], (-1, 1))
        lap = x[..., ip1, :] - 2.0 * x + x[..., im1, :]
        return self.local(x) + self.eps * lap @ self.H.T

    def jac_blocks(self, x):
        n = x.shape[-2]
        shape = x.shape[:-2] + (n, self.k, self.k)
        C = np.broadcast_to(self.eps * self.H, shape)
        D = self.local_jac(x) - 2.0 * self.eps * self.H
        return np.stack([C, D, C], axis=-4)


class LorenzRing(DiffusiveRing):
    name = "Lorenz"

    def __init__(self, eps, sigma=10.0, rho=28.0, beta=8.0 / 3.0, H=None):
        super().__init__(eps, np.diag([1.0, 0.0, 0.0]) if H is None else H)
        self.sigma, self.rho, self.beta = sigma, rho, beta

    def local(self, x):
        a, b, c = x[..., 0], x[..., 1], x[..., 2]
        return np.stack([self.sigma * (b - a), a * (self.rho - c) - b, a * b - self.beta * c], axis=-1)

    def local_jac(self, x):
        a, b, c = x[..., 0], x[..., 1], x[..., 2]
        z = np.zeros_like(a)
        o = np.ones_like(a)
        r0 = np.stack([-self.sigma * o, self.sigma * o, z], axis=-1)
        r1 = np.stack([self.rho - c, -o, -a], axis=-1)
        r2 = np.stack([b, a, -self.beta * o], axis=-1)
        return np.stack([r0, r1, r2], axis=-2)


class RosslerRing(DiffusiveRing):
    name = "Rossler"

    def __init__(self, eps, a=0.2, b=0.2, c=5.7, H=None):
        super().__init__(eps, np.diag([1.0, 0.0, 0.0]) if H is None else H)
        self.a, self.b, self.c = a, b, c

    def local(self, x):
        u, v, w = x[..., 0], x[..., 1], x[..., 2]
        return np.stack([-v - w, u + self.a * v, self.b + w * (u - self.c)], axis=-1)

    def local_jac(self, x):
        u, v, w = x[..., 0], x[..., 1], x[..., 2]
        z = np.zeros_like(u)
        o = np.ones_like(u)
        r0 = np.stack([z, -o, -o], axis=-1)
        r1 = np.stack([o, self.a * o, z], axis=-1)
        r2 = np.stack([w, z, u - self.c], axis=-1)
        return np.stack([r0, r1, r2], axis=-2)


class HindmarshRoseRing(DiffusiveRing):
    name = "Hindmarsh-Rose"

    def __init__(self, eps, a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, xr=-1.6, I=3.25, H=None):
        super().__init__(eps, np.diag([1.0, 0.0, 0.0]) if H is None else H)
        self.a, self.b, self.c, self.d, self.r, self.s, self.xr, self.I = a, b, c, d, r, s, xr, I

    def local(self, x):
        u, v, w = x[..., 0], x[..., 1], x[..., 2]
        du = v - self.a * u ** 3 + self.b * u ** 2 - w + self.I
        dv = self.c - self.d * u ** 2 - v
        dw = self.r * (self.s * (u - self.xr) - w)
        return np.stack([du, dv, dw], axis=-1)

    def local_jac(self, x):
        u = x[..., 0]
        z = np.zeros_like(u)
        o = np.ones_like(u)
        r0 = np.stack([-3.0 * self.a * u ** 2 + 2.0 * self.b * u, o, -o], axis=-1)
        r1 = np.stack([-2.0 * self.d * u, -o, z], axis=-1)
        r2 = np.stack([self.r * self.s * o, z, -self.r * o], axis=-1)
        return np.stack([r0, r1, r2], axis=-2)
