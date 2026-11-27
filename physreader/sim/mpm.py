"""MLS-MPM in NVIDIA Warp: Neo-Hookean elasticity, von Mises plasticity and frictional walls.

`repaired=False` is the solver that generated PhysReader-Sim. `repaired=True` is the version used for
the real-drop replays and for the simulations marked "repaired" in PhysReader-Sim: a radial return that
lands on the yield surface and is applied as an isochoric increment, a velocity gradient from the
B-spline weight gradients, compensated position updates, and walls that let separating nodes go.
"""
import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False})

# Material constants are formed in float32 from E in units of 1e7 Pa and yield stress in 1e5 Pa.
E_UNIT = 1e7
YS_UNIT = 1e5


@wp.struct
class Particles:
    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    C: wp.array(dtype=wp.mat33)
    F: wp.array(dtype=wp.mat33)
    F_trial: wp.array(dtype=wp.mat33)
    stress: wp.array(dtype=wp.mat33)
    vol: wp.array(dtype=float)
    mass: wp.array(dtype=float)
    x_err: wp.array(dtype=wp.vec3)


@wp.struct
class Grid:
    n: int
    dx: float
    inv_dx: float
    m: wp.array(dtype=float, ndim=3)
    mv: wp.array(dtype=wp.vec3, ndim=3)
    v: wp.array(dtype=wp.vec3, ndim=3)
    force: wp.array(dtype=wp.vec3, ndim=3)
    wall: wp.array(dtype=int, ndim=3)


@wp.kernel
def material_constants(E: float, nu: float, yield_stress: float, out: wp.array(dtype=float)):
    out[0] = E_UNIT * E / (2.0 * (1.0 + nu))
    out[1] = E_UNIT * E * nu / ((1.0 + nu) * wp.max(1.0 - 2.0 * nu, 1e-8))
    out[2] = YS_UNIT * yield_stress


@wp.func
def exp_diag(h: wp.vec3):
    return wp.diag(wp.vec3(wp.exp(h[0]), wp.exp(h[1]), wp.exp(h[2])))


@wp.func
def von_mises(F: wp.mat33, mu: float, lam: float, yield_stress: float, repaired: int):
    J = wp.determinant(F)
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)
    strain = wp.vec3(wp.log(wp.max(sig[0], 1e-4)), wp.log(wp.max(sig[1], 1e-4)), wp.log(wp.max(sig[2], 1e-4)))
    mean = (strain[0] + strain[1] + strain[2]) / 3.0
    tau = 2.0 * mu * strain + lam * (strain[0] + strain[1] + strain[2]) * wp.vec3(1.0, 1.0, 1.0)
    tau_sum = tau[0] + tau[1] + tau[2]
    s = wp.vec3(tau[0] - tau_sum / 3.0, tau[1] - tau_sum / 3.0, tau[2] - tau_sum / 3.0)
    tau_eq = wp.sqrt(3.0 / 2.0 * wp.dot(s, s))
    tau_y = yield_stress * J
    if tau_eq > tau_y:
        dev = strain - wp.vec3(mean, mean, mean)
        if repaired == 0:
            gamma = (tau_eq - tau_y) / (2.0 * mu)
            return U * exp_diag(strain - (gamma / (wp.length(dev) + 1e-6)) * dev) * wp.transpose(V)
        norm = wp.max(wp.length(dev), 1e-12)
        gamma = (tau_eq - tau_y) / (wp.sqrt(6.0) * mu)
        strain = strain - (gamma / norm) * dev
        eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_elastic = F + (F * V) * (exp_diag(-(gamma / norm) * dev) - eye) * wp.transpose(V)
        if J <= 0.0 or sig[0] < 1e-4 or sig[1] < 1e-4 or sig[2] < 1e-4:
            F_elastic = U * exp_diag(strain) * wp.transpose(V)
        return F_elastic
    return F


@wp.kernel
def compute_stress(p: Particles, mu: float, lam: float, yield_stress: float, repaired: int):
    i = wp.tid()
    F = von_mises(p.F_trial[i], mu, lam, yield_stress, repaired)
    p.F[i] = F
    J = wp.max(wp.determinant(F), 1e-8)
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tau = mu * (F * wp.transpose(F) - eye) + lam * wp.log(J) * eye
    p.stress[i] = (tau + wp.transpose(tau)) / 2.0


@wp.func
def base_cell(x: wp.vec3, inv_dx: float):
    q = x * inv_dx
    bx = wp.int(q[0] - 0.5)
    by = wp.int(q[1] - 0.5)
    bz = wp.int(q[2] - 0.5)
    return bx, by, bz, q - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))


@wp.func
def bspline(fx: wp.vec3):
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    return wp.mat33(wp.cw_mul(wa, wa) * 0.5, wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                    wp.cw_mul(wc, wc) * 0.5)


@wp.func
def bspline_slope(fx: wp.vec3):
    return wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))


@wp.func
def weight_grad(w: wp.mat33, dw: wp.mat33, a: int, b: int, c: int, inv_dx: float):
    grad = wp.vec3(dw[0, a] * w[1, b] * w[2, c], w[0, a] * dw[1, b] * w[2, c], w[0, a] * w[1, b] * dw[2, c])
    return grad * inv_dx


@wp.kernel
def p2g(p: Particles, g: Grid):
    i = wp.tid()
    stress = p.stress[i]
    bx, by, bz, fx = base_cell(p.x[i], g.inv_dx)
    w = bspline(fx)
    dw = bspline_slope(fx)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                dpos = (wp.vec3(wp.float(a), wp.float(b), wp.float(c)) - fx) * g.dx
                ix = wp.clamp(bx + a, 0, g.n - 1)
                iy = wp.clamp(by + b, 0, g.n - 1)
                iz = wp.clamp(bz + c, 0, g.n - 1)
                weight = w[0, a] * w[1, b] * w[2, c]
                force = -p.vol[i] * stress * weight_grad(w, dw, a, b, c, g.inv_dx)
                momentum = weight * p.mass[i] * (p.v[i] + p.C[i] * dpos)
                wp.atomic_add(g.mv, ix, iy, iz, momentum)
                wp.atomic_add(g.m, ix, iy, iz, weight * p.mass[i])
                wp.atomic_add(g.force, ix, iy, iz, force)


@wp.kernel
def grid_update(g: Grid, gravity: wp.vec3, dt: float):
    i, j, k = wp.tid()
    m = g.m[i, j, k]
    if m > 1e-15:
        v = g.mv[i, j, k] * (1.0 / m)
        f = g.force[i, j, k] + m * gravity
        g.v[i, j, k] = v + dt * f / m


@wp.kernel
def collide(g: Grid, point: wp.vec3, normal: wp.vec3, friction: float, repaired: int):
    i, j, k = wp.tid()
    offset = wp.vec3(float(i) * g.dx - point[0], float(j) * g.dx - point[1], float(k) * g.dx - point[2])
    if wp.dot(offset, normal) < 0.0:
        g.wall[i, j, k] = 1
        v = g.v[i, j, k]
        vn = wp.dot(v, normal)
        if vn > 0.0:
            if repaired == 0:
                g.v[i, j, k] = wp.vec3((1.0 - friction) * v[0], 0.0, (1.0 - friction) * v[2])
        else:
            vt = v - vn * normal
            vt_norm = wp.length(vt)
            if vn < 0.0 and vt_norm > 1e-4 and friction > 0.0:
                vt_new = wp.max(0.0, vt_norm + vn * friction)
                if vt_new < 1e-4:
                    vt = wp.vec3(0.0)
                else:
                    vt = vt * wp.min(1.0, wp.max(0.0, vt_new / vt_norm))
            g.v[i, j, k] = vt


@wp.func
def buried(wall: wp.array(dtype=int, ndim=3), i: int, j: int, k: int):
    if j < 4:
        return wall[i, j, k] == 1 and wall[i, j + 1, k] == 1
    return False


@wp.func
def affine_velocity(D: wp.mat33, B: wp.mat33, dx: float):
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    dxx = wp.max(D[0, 0], 0.0) + 1e-8
    dyy = wp.max(D[1, 1], 0.0) + 1e-8
    dzz = wp.max(D[2, 2], 0.0) + 1e-8
    sx = wp.clamp(1.0 / wp.sqrt(dxx), 0.0, 10.0)
    sy = wp.clamp(1.0 / wp.sqrt(dyy), 0.0, 10.0)
    sz = wp.clamp(1.0 / wp.sqrt(dzz), 0.0, 10.0)
    trace = D[0, 0] + D[1, 1] + D[2, 2] + 1e-8
    if dyy / trace < 0.2:
        M = wp.mat33(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, sz)
        Dm = M * D * M
        lam = wp.max(1e-4 * dx * dx, 1e-3 * (Dm[0, 0] + Dm[1, 1] + Dm[2, 2]) / 3.0)
        return B * M * wp.inverse(Dm + eye * lam) * M
    lam = wp.max(1e-12, 1e-3 * trace / 3.0)
    return B * wp.inverse(D + eye * lam)


@wp.kernel
def g2p(p: Particles, g: Grid, dt: float, repaired: int):
    i = wp.tid()
    bx, by, bz, fx = base_cell(p.x[i], g.inv_dx)
    w = bspline(fx)
    dw = bspline_slope(fx)
    total = float(0.0)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                ix = bx + a
                iy = by + b
                iz = bz + c
                if ix < 0 or ix >= g.n or iy < 0 or iy >= g.n or iz < 0 or iz >= g.n:
                    continue
                if not buried(g.wall, ix, iy, iz):
                    total = total + w[0, a] * w[1, b] * w[2, c]
    v_mean = wp.vec3(0.0)
    r_mean = wp.vec3(0.0)
    D = wp.mat33(0.0)
    B = wp.mat33(0.0)
    L = wp.mat33(0.0)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                ix = bx + a
                iy = by + b
                iz = bz + c
                if ix < 0 or ix >= g.n or iy < 0 or iy >= g.n or iz < 0 or iz >= g.n:
                    continue
                dpos = wp.vec3(wp.float(a), wp.float(b), wp.float(c)) - fx
                weight = w[0, a] * w[1, b] * w[2, c] / total
                v = g.v[ix, iy, iz]
                L = L + wp.outer(v, weight_grad(w, dw, a, b, c, g.inv_dx))
                if not buried(g.wall, ix, iy, iz):
                    r = dpos * g.dx
                    v_mean = v_mean + v * weight
                    r_mean = r_mean + r * weight
                    D = D + wp.outer(r, r) * weight
                    B = B + wp.outer(v, r) * weight
    G = affine_velocity(D - wp.outer(r_mean, r_mean), B - wp.outer(v_mean, r_mean), g.dx)
    if repaired == 0:
        L = G
    p.v[i] = v_mean
    x = p.x[i]
    if repaired == 0:
        p.x[i] = x + dt * v_mean
    else:
        step = dt * v_mean - p.x_err[i]
        x_new = x + step
        p.x_err[i] = (x_new - x) - step
        p.x[i] = x_new
    p.C[i] = G
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    F = p.F[i]
    F_half = (eye + 0.5 * dt * L) * F
    p.F_trial[i] = F + dt * (L * F_half)


class Solver:
    """Particles in the cube [0, grid_lim]^3 with n_grid cells per axis.
    E and yield_stress in Pa, density in kg/m^3, SI units throughout.
    `velocity` is one vector for all particles or one per particle."""

    def __init__(self, positions, volumes, E, nu, density, yield_stress, velocity=(0.0, 0.0, 0.0),
                 n_grid=80, grid_lim=2.0, gravity=(0.0, -9.8, 0.0), repaired=False, device='cuda:0'):
        wp.init()
        self.device = device
        self.repaired = int(repaired)
        self.gravity = wp.vec3(*gravity)
        self.walls = []
        n = len(positions)
        vol = np.asarray(volumes, np.float32)
        p = Particles()
        p.x = self._array(positions, wp.vec3)
        p.v = self._array(np.broadcast_to(np.asarray(velocity, np.float32), (n, 3)).copy(), wp.vec3)
        p.C = wp.zeros(n, dtype=wp.mat33, device=device)
        p.F = wp.zeros(n, dtype=wp.mat33, device=device)
        p.F_trial = self._array(np.tile(np.eye(3), (n, 1, 1)), wp.mat33)
        p.stress = wp.zeros(n, dtype=wp.mat33, device=device)
        p.vol = self._array(vol, float)
        p.mass = self._array(np.float32(density) * vol, float)
        p.x_err = wp.zeros(n, dtype=wp.vec3, device=device)
        g = Grid()
        g.n = n_grid
        g.dx = grid_lim / n_grid
        g.inv_dx = n_grid / grid_lim
        shape = (n_grid, n_grid, n_grid)
        g.m = wp.zeros(shape, dtype=float, device=device)
        g.mv = wp.zeros(shape, dtype=wp.vec3, device=device)
        g.v = wp.zeros(shape, dtype=wp.vec3, device=device)
        g.force = wp.zeros(shape, dtype=wp.vec3, device=device)
        g.wall = wp.zeros(shape, dtype=int, device=device)
        self.particles, self.grid, self.n, self.shape = p, g, n, shape
        out = wp.zeros(3, dtype=float, device=device)
        wp.launch(material_constants, 1, [E / E_UNIT, nu, yield_stress / YS_UNIT, out], device=device)
        self.mu, self.lam, self.yield_stress = (float(c) for c in out.numpy())

    def _array(self, values, dtype):
        return wp.array(np.asarray(values, np.float32), dtype=dtype, device=self.device)

    def add_wall(self, point, normal, friction):
        """Half-space wall through `point`; grid nodes behind it (against `normal`) are constrained."""
        normal = np.asarray(normal, np.float64) / np.linalg.norm(normal)
        self.walls.append((wp.vec3(*point), wp.vec3(*normal), float(friction)))

    def step(self, dt):
        p, g, device = self.particles, self.grid, self.device
        for a in (g.m, g.mv, g.v, g.force, g.wall):
            a.zero_()
        wp.launch(compute_stress, self.n, [p, self.mu, self.lam, self.yield_stress, self.repaired],
                  device=device)
        wp.launch(p2g, self.n, [p, g], device=device)
        wp.launch(grid_update, self.shape, [g, self.gravity, dt], device=device)
        for point, normal, friction in self.walls:
            wp.launch(collide, self.shape, [g, point, normal, friction, self.repaired], device=device)
        wp.launch(g2p, self.n, [p, g, dt, self.repaired], device=device)

    def positions(self):
        return self.particles.x.numpy()
