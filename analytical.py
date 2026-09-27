"""
Analytical equilibrium solutions of the generalized two-player game,
following analytical.md.

Everything is expressed in normalized units: rewards are divided by R, so
p = P/R and utilities are u/R (u_total = 2 * value). Z = 0 throughout.
Actions are failure probabilities in [0, 1].

Phase diagram (p = P/R):
  tau = 0                          atomless density on [0, h]      solve_tau0
  tau >= tau0(p) = 1/(2(2p+1))     pure equilibrium at 0           solve_frictional
  tau_c(p) <= tau < tau0(p)        pure equilibrium at r_p(tau)    solve_frictional
  0 < tau < tau_c(p)               finite atomic mixture           solve_frictional
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, root
from scipy.special import expit
from scipy.stats import norm

from game import joint_failure_probability

_CLIP = 1e-14


def C_rho(a, b, rho):
    """Joint failure probability C_rho(a, b) (Gaussian copula), vectorized."""
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    out = joint_failure_probability(
        np.ascontiguousarray(a).ravel(), np.ascontiguousarray(b).ravel(), float(rho)
    )
    return out.reshape(a.shape)


def q_rho(a, b, rho):
    """dC_rho/da (eq. 7), the conditional CDF Phi((z_b - rho z_a)/sqrt(1-rho^2)).

    Only defined for -1 < rho < 1 (the copula has a kink at rho = +-1).
    """
    if not -1.0 < rho < 1.0:
        raise ValueError("q_rho requires -1 < rho < 1")
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    if rho == 0.0:
        return b.copy()
    za = norm.ppf(np.clip(a, _CLIP, 1 - _CLIP))
    zb = norm.ppf(np.clip(b, _CLIP, 1 - _CLIP))
    return norm.cdf((zb - rho * za) / np.sqrt(1.0 - rho**2))


def D_rho(a, rho):
    """Probability that both players survive at (a, a) (eq. 6).

    Computed as C_rho(1-a, 1-a) (the Gaussian copula is radially symmetric),
    which keeps full relative precision as a -> 1, unlike 1 - 2a + C_rho(a, a).
    """
    b = 1.0 - np.asarray(a, dtype=np.float64)
    return C_rho(b, b, rho)


def U(a, b, p, tau, rho):
    """Normalized payoff of the a-player against the b-player (eq. 1).

    tau = 0 uses the tie value s = 1/2, matching game.reward with noise=0.
    """
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    c = C_rho(a, b, rho)
    if tau == 0.0:
        s = 0.5 * (np.sign(a - b) + 1.0)
    else:
        s = expit((a - b) / tau)
    return b - c - p * a + (1.0 - a - b + c) * s


def U_a(a, b, p, tau, rho):
    """Own-action derivative dU/da (eq. 44). Requires tau > 0."""
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    s = expit((a - b) / tau)
    ca = b.copy() if rho == 0.0 else q_rho(a, b, rho)
    c = C_rho(a, b, rho)
    return -p - s - (1.0 - s) * ca + (1.0 - a - b + c) * s * (1.0 - s) / tau


# ---------------------------------------------------------------------------
# Phase thresholds (eqs. 26, 33, 34)
# ---------------------------------------------------------------------------


def tau0(p):
    """Zero-risk dominance threshold (eq. 26), independent of rho."""
    return 1.0 / (2.0 * (2.0 * p + 1.0))


def _eta(x):
    return 2.0 * np.tanh(x / 2.0) / x


def tau_c(p):
    """Critical friction below which the equilibrium is mixed (eqs. 33-34), rho=0."""

    def g(x):
        e = _eta(x)
        return x * e**2 / (4.0 * (1.0 - e)) - 1.0 + e / 2.0 - p

    x_c = brentq(g, 1e-3, 500.0, xtol=1e-13)
    return (1.0 - _eta(x_c)) / x_c


def pure_action(p, tau, rho=0.0):
    """Symmetric pure equilibrium candidate (eq. 30 for rho=0, eq. 45 otherwise)."""
    if tau >= tau0(p):
        return 0.0
    if rho == 0.0:
        return 1.0 + tau - np.sqrt(tau**2 + 4.0 * tau * (p + 1.0))

    def g(r):
        return float(D_rho(r, rho)) - 2.0 * tau * (
            2.0 * p + 1.0 + float(q_rho(r, r, rho))
        )

    return brentq(g, 1e-12, 1.0 - 1e-9, xtol=1e-14)


# ---------------------------------------------------------------------------
# Equilibrium container
# ---------------------------------------------------------------------------


@dataclass
class Equilibrium:
    p: float
    tau: float
    rho: float
    kind: str  # "continuous" or "atomic"
    mean: float
    value: float  # per-player normalized utility; u_total = 2 * value
    # continuous equilibria (tau = 0)
    grid: np.ndarray = None
    density: np.ndarray = None
    h: float = None
    # atomic equilibria (tau > 0)
    atoms: np.ndarray = None
    weights: np.ndarray = None
    check: dict = field(default_factory=dict)

    @property
    def u_total(self):
        return 2.0 * self.value

    def cdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.kind == "continuous":
            F = cumulative_trapezoid(self.density, self.grid, initial=0.0)
            F /= F[-1]
            return np.interp(x, self.grid, F, left=0.0, right=1.0)
        return (self.weights[None, :] * (self.atoms[None, :] <= x.reshape(-1, 1))).sum(
            axis=1
        ).reshape(x.shape)


# ---------------------------------------------------------------------------
# tau = 0: atomless equilibrium on [0, h] (section 1)
# ---------------------------------------------------------------------------


def _tau0_kernel(p, rho, h, n):
    """Nystrom (trapezoid) discretization of K_{rho,h} (eq. 9) on [0, h].

    The integral is split exactly at the diagonal node, so the composite
    trapezoid rule keeps O(1/n^2) accuracy despite the kernel jump.
    """
    a = np.linspace(0.0, h, n)
    da = h / (n - 1)
    Q = q_rho(a[:, None], a[None, :], rho)
    B = np.where(a[None, :] < a[:, None], 1.0, Q)
    np.fill_diagonal(B, 0.5 * (1.0 + np.diag(Q)))
    w = np.full(n, da)
    w[0] = w[-1] = da / 2.0
    M = B * w
    M[0, 0] = 0.5 * da * Q[0, 0]
    M[-1, -1] = 0.5 * da
    Dv = D_rho(a, rho)
    return a, M / Dv[:, None], Dv


def _resolvent_density(p, rho, h, n):
    """Solve (I - K) f = p / D on [0, h]; returns (a, f, mass) or None past the
    first Fredholm singularity."""
    a, K, Dv = _tau0_kernel(p, rho, h, n)
    try:
        f = np.linalg.solve(np.eye(n) - K, p / Dv)
    except np.linalg.LinAlgError:
        return None
    if np.any(f < -1e-9):
        return None
    return a, f, np.trapezoid(f, a)


class _Perron:
    """Spectral radius / eigenfunction of K via power iteration, warm-started."""

    def __init__(self):
        self.v = None

    def __call__(self, K):
        n = K.shape[0]
        v = np.ones(n) / n if self.v is None or len(self.v) != n else self.v
        lam = 0.0
        for _ in range(5000):
            w = K @ v
            lam_new = np.linalg.norm(w)
            v = w / lam_new
            if abs(lam_new - lam) < 1e-14 * lam_new:
                break
            lam = lam_new
        self.v = v
        return lam_new, np.abs(v)


def _tau0_indifference_residual(a, f, p, rho):
    """max_a |V(a) - mu| over the support: indifference check for eq. (4)."""
    F = cumulative_trapezoid(f, a, initial=0.0)
    mu = np.trapezoid(a * f, a)
    Cm = C_rho(a[:, None], a[None, :], rho)
    resid = np.empty(len(a))
    for i in range(len(a)):
        integrand = (a[i:] - Cm[i, i:] - p * a[i]) * f[i:]
        Vi = (1.0 - (p + 1.0) * a[i]) * F[i] + np.trapezoid(integrand, a[i:])
        resid[i] = Vi - mu
    return float(np.max(np.abs(resid)))


def _shooting_march(p, rho, h, n, grading=2.0):
    """March the backward Volterra form of eq. (8) from F(h) = 1 down to x = 0.

    Integrating (8) by parts with F(h) = 1 gives
        D(x) F'(x) = p + q(x, h) + [1 - q(x, x)] F(x) - int_x^h dq(x, y)/dy F(y) dy,
    whose right side only uses F on [x, h]. Each step is a Heun
    predictor-corrector step. The integral uses the exact increments of q over
    each cell times the cell mean of F, so no q_y evaluation is needed. The
    grid x_i = h (i/n)^grading is finer near 0, where f has a power-law term
    q(x, x) ~ x^((1-rho)/(1+rho)); this keeps the error at O(1/n^2) for all rho.

    Returns (x, F, f) with f = F'.
    """
    x = h * np.linspace(0.0, 1.0, n + 1) ** grading
    dx = np.diff(x)
    if rho == 0.0:
        Q = np.broadcast_to(x[None, :], (n + 1, n + 1))
    else:
        z = norm.ppf(np.clip(x, _CLIP, 1 - _CLIP))
        Q = norm.cdf((z[None, :] - rho * z[:, None]) / np.sqrt(1.0 - rho**2))
    dQ = np.diff(Q, axis=1)  # dQ[i, j] = q(x_i, x_{j+1}) - q(x_i, x_j)
    Dv = D_rho(x, rho)
    q_h, q_diag = Q[:, -1], np.diag(Q)
    F = np.zeros(n + 1)
    F[-1] = 1.0
    F_cell = np.zeros(n)
    f = np.zeros(n + 1)

    def slope(i):
        J = dQ[i, i:] @ F_cell[i:]
        return (p + q_h[i] + (1.0 - q_diag[i]) * F[i] - J) / Dv[i]

    f[n] = slope(n)
    for i in range(n - 1, -1, -1):
        F[i] = F[i + 1] - dx[i] * f[i + 1]  # predictor (backward Euler)
        F_cell[i] = 0.5 * (F[i] + F[i + 1])
        f_pred = slope(i)
        F[i] = F[i + 1] - 0.5 * dx[i] * (f[i + 1] + f_pred)  # corrector
        F_cell[i] = 0.5 * (F[i] + F[i + 1])
        f[i] = slope(i)
    return x, F, f


def _solve_tau0_shooting(p, rho, n):
    """Support endpoint h as the root of R(h) = F_h(0), and the density on [0, h]."""
    lo, hi = 1e-3, (1.0 - 1e-9) / (p + 1.0)  # eq. (5) with mu > 0 gives h < 1/(p+1)

    def R(h):
        return _shooting_march(p, rho, h, n)[1][0]

    if not R(lo) > 0.0 > R(hi):
        raise RuntimeError("F_h(0) does not change sign on (0, 1/(p+1))")
    h = brentq(R, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps)
    x, F, f = _shooting_march(p, rho, h, n)
    return h, x, F, f


def solve_tau0(p, rho, n=2000, closed_form=True, method="shooting"):
    """Atomless tau=0 equilibrium density on [0, h] (eqs. 8-18).

    Uses the closed forms (13)-(18) for rho in {0, 1, -1} when closed_form is
    True. Otherwise method="shooting" marches the backward Volterra form of (8)
    from F(h) = 1 and selects h by F(0) = 0 (error O(1/n^2) for every rho), and
    method="fredholm" uses the resolvent (10)-(11) (p > 0) or the Perron
    eigenfunction (12) (p = 0).
    """
    if closed_form and rho == 0.0:
        k = np.sqrt((p + 1.0) ** 2 + 1.0)
        h = (p + 2.0 - k) / (p + 1.0)
        a = np.linspace(0.0, h, n)
        f = (k - 1.0) / (1.0 - a) ** 3
        mu = k - p - 1.0
    elif closed_form and rho == 1.0:
        h = 1.0 - np.exp(-1.0 / (p + 1.0))
        a = np.linspace(0.0, h, n)
        f = (p + 1.0) / (1.0 - a)
        mu = (p + 1.0) * np.exp(-1.0 / (p + 1.0)) - p
    elif closed_form and rho == -1.0:
        if p <= 0.0:
            raise ValueError("rho = -1 closed form requires p > 0")
        h = (2.0 * p + 1.0) / (2.0 * (p + 1.0) ** 2)
        a = np.linspace(0.0, h, n)
        f = p * (1.0 - 2.0 * a) ** (-1.5)
        mu = 1.0 / (2.0 * (p + 1.0))
    elif method == "shooting":
        if not -1.0 < rho < 1.0:
            raise ValueError("the shooting solver requires -1 < rho < 1")
        h, a, F, f = _solve_tau0_shooting(p, rho, n)
        # int x dF = h F(h) - int F; end-corrected trapezoid since F' = f is known
        da = np.diff(a)
        mu = h - np.sum(0.5 * da * (F[:-1] + F[1:]) + da**2 / 12.0 * (f[:-1] - f[1:]))
    elif method == "fredholm":
        if not -1.0 < rho < 1.0:
            raise ValueError("the Fredholm solver requires -1 < rho < 1")
        h = _solve_tau0_endpoint(p, rho)
        if p > 0.0:
            a, f, mass = _resolvent_density(p, rho, h, n)
            f = f / mass  # residual normalization (mass = 1 up to brentq tol)
        else:
            _, K, _ = _tau0_kernel(p, rho, h, n)
            _, f = _Perron()(K)
            a = np.linspace(0.0, h, n)
            f = f / np.trapezoid(f, a)
        mu = float(np.trapezoid(a * f, a))
    else:
        raise ValueError(f"unknown method {method!r}")

    check = {
        "indifference": _tau0_indifference_residual(a, f, p, rho),
        "endpoint_identity": abs(mu - (1.0 - (p + 1.0) * h)),  # eq. (5)
    }
    return Equilibrium(
        p=p, tau=0.0, rho=rho, kind="continuous",
        mean=float(mu), value=float(mu),  # eq. (19): v = mu
        grid=a, density=f, h=float(h), check=check,
    )


def _solve_tau0_endpoint(p, rho, n_scan=400, n_solve=1200):
    """Support endpoint h: mass = 1 (eq. 11) for p > 0, r(K) = 1 (eq. 12) for p = 0."""
    if p > 0.0:

        def g(h, n):
            out = _resolvent_density(p, rho, h, n)
            return np.inf if out is None else out[2] - 1.0

    else:
        perron = _Perron()

        def g(h, n):
            _, K, _ = _tau0_kernel(p, rho, h, n)
            return perron(K)[0] - 1.0

    lo = 0.02
    if g(lo, n_scan) >= 0:
        raise RuntimeError("bracketing failed at the lower end")
    hi = None
    for h in np.arange(lo + 0.01, 0.999, 0.01):
        val = g(h, n_scan)
        if np.isfinite(val) and val >= 0:
            hi = h
            break
        if not np.isfinite(val):
            raise RuntimeError("hit a Fredholm singularity before mass reached 1")
        lo = h
    if hi is None:
        raise RuntimeError("no endpoint found in (0, 1)")
    # refine on the fine grid; pad the coarse bracket to be safe
    lo, hi = max(lo - 0.01, 1e-3), min(hi + 0.01, 0.999)
    return brentq(lambda h: g(h, n_solve), lo, hi, xtol=1e-10)


# ---------------------------------------------------------------------------
# tau > 0: pure and finite atomic equilibria (sections 2-6)
# ---------------------------------------------------------------------------


def best_response_gap(atoms, weights, p, tau, rho, ngrid=100001):
    """max_a V(a) - v for the mixture (atoms, weights): global condition (25)/(46).

    Returns (gap, argmax, v).
    """
    atoms = np.asarray(atoms, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, ngrid)
    Vg = U(grid[:, None], atoms[None, :], p, tau, rho) @ weights
    v = float(weights @ (U(atoms[:, None], atoms[None, :], p, tau, rho) @ weights))
    i = int(np.argmax(Vg))
    return float(Vg[i] - v), float(grid[i]), v


def _two_atom_candidates(p, tau, rho):
    """Solutions x of the scalar stationarity equation (41) with 0 < alpha < 1."""

    def pieces(x):
        x = np.asarray(x, dtype=np.float64)
        c = C_rho(x, x, rho)
        ell = expit(x / tau)
        alpha = (1.0 + c + 2.0 * p * x - 2.0 * (1.0 - x) * ell) / c  # eq. (38)
        dx0 = -p - ell + (1.0 - x) * ell * (1.0 - ell) / tau  # eq. (39)
        q = x.copy() if rho == 0.0 else q_rho(x, x, rho)
        d = 1.0 - 2.0 * x + c
        dxx = -p - 0.5 - 0.5 * q + d / (4.0 * tau)  # eq. (40)
        return alpha, alpha * dx0 + (1.0 - alpha) * dxx  # eq. (41)

    xs = np.linspace(1e-4, 0.995, 4000)
    _, g = pieces(xs)
    out = []
    for i in np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]:
        x = brentq(lambda t: float(pieces(t)[1]), xs[i], xs[i + 1], xtol=1e-14)
        alpha = float(pieces(x)[0])
        if 0.0 < alpha < 1.0:
            out.append((x, alpha))
    return out


def _solve_atom_system(p, tau, rho, x_init, w_init, v_init):
    """Solve the finite-support system (23)-(24) for a symmetric equilibrium
    whose support contains 0 (nodes x, weights w, value v)."""
    m = len(x_init)
    assert x_init[0] == 0.0

    def unpack(z):
        return np.concatenate([[0.0], z[: m - 1]]), z[m - 1 : 2 * m - 1], z[2 * m - 1]

    def eqs(z):
        x, w, v = unpack(z)
        A = U(x[:, None], x[None, :], p, tau, rho)
        B = U_a(x[:, None], x[None, :], p, tau, rho)
        return np.concatenate([A @ w - v, (B @ w)[1:], [w.sum() - 1.0]])

    z0 = np.concatenate([x_init[1:], w_init, [v_init]])
    sol = root(eqs, z0, method="hybr", tol=1e-13)
    x, w, v = unpack(sol.x)
    ok = (
        sol.success
        and np.all(np.diff(x) > 1e-10)
        and np.all(w > 0)
        and np.all(x >= 0)
        and x[-1] <= 1
    )
    return x, w, float(v), ok, float(np.max(np.abs(eqs(sol.x))))


def solve_frictional(p, tau, rho=0.0, gap_tol=1e-7, max_atoms=8, ngrid=100001):
    """Equilibrium for tau > 0: pure delta_0, pure delta_r, or finite atomic
    mixture, following the phase diagram and eqs. (23)-(42)."""
    assert tau > 0.0

    def atomic(atoms, weights, v, extra):
        atoms = np.asarray(atoms, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        gap, argmax, v_grid = best_response_gap(atoms, weights, p, tau, rho, ngrid)
        check = {"br_gap": gap, "br_argmax": argmax, **extra}
        return Equilibrium(
            p=p, tau=tau, rho=rho, kind="atomic",
            mean=float(weights @ atoms), value=v,
            atoms=atoms, weights=weights, check=check,
        ), gap

    # zero-risk phase (eq. 28)
    if tau >= tau0(p):
        eq, gap = atomic([0.0], [1.0], 0.5, {"phase": "zero-risk pure"})
        if gap < gap_tol:
            return eq

    # positive pure phase (eqs. 30/45 + global check 32/46)
    r = pure_action(p, tau, rho)
    if r > 0.0:
        eq, gap = atomic([r], [1.0], float(U(r, r, p, tau, rho)), {"phase": "positive pure"})
        if gap < gap_tol:
            return eq

    # two-atom phase (eqs. 36-42)
    best = None
    for x, alpha in _two_atom_candidates(p, tau, rho):
        v = alpha * 0.5 + (1.0 - alpha) * float(U(0.0, x, p, tau, rho))
        eq, gap = atomic([0.0, x], [alpha, 1.0 - alpha], v, {"phase": "2-atom mixed"})
        if gap < gap_tol:
            return eq
        if best is None or gap < best[1]:
            best = (eq, gap)

    if best is None:
        raise RuntimeError("no two-atom candidate found to grow the support from")

    # grow the support one atom at a time (eq. 43 + system 23-25)
    eq, gap = best
    x, w, v = eq.atoms, eq.weights, eq.value
    while len(x) < max_atoms:
        y = eq.check["br_argmax"]
        order = np.argsort(np.append(x, y))
        x_init = np.append(x, y)[order]
        w_init = np.append(w * (1.0 - 0.02), 0.02)[order]
        x, w, v, ok, resid = _solve_atom_system(p, tau, rho, x_init, w_init, v)
        if not ok:
            raise RuntimeError(f"{len(x_init)}-atom system did not converge")
        eq, gap = atomic(x, w, v, {"phase": f"{len(x)}-atom mixed", "system_residual": resid})
        if gap < gap_tol:
            return eq
    raise RuntimeError(f"no equilibrium found with up to {max_atoms} atoms")


def solve(p, tau, rho=0.0, **kwargs):
    """Analytical equilibrium for any (p, tau, rho) in the covered phases."""
    if tau == 0.0:
        return solve_tau0(p, rho, **kwargs)
    return solve_frictional(p, tau, rho, **kwargs)


# ---------------------------------------------------------------------------
# Self-checks against the reference values quoted in analytical.md
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings

    # FP flags raised inside numba/cython code get attributed by numpy to the
    # next checked operation; matmuls of finite arrays cannot raise these.
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul", category=RuntimeWarning
    )

    rng = np.random.default_rng(0)

    # payoff identity (2): U(a,b) + U(b,a) = 1 - C - p(a+b), friction cancels
    a, b = rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)
    for rho, tau, p in [(0.0, 0.1, 1.0), (0.5, 0.03, 0.0), (-0.7, 0.2, 2.0)]:
        lhs = U(a, b, p, tau, rho) + U(b, a, p, tau, rho)
        rhs = 1.0 - C_rho(a, b, rho) - p * (a + b)
        assert np.allclose(lhs, rhs, atol=1e-12), (rho, tau, p)
    print("identity (2): ok")

    # U matches the repository implementation game.reward (R=1, Z=0)
    from game import reward

    for rho, tau, p in [(0.0, 0.1, 1.0), (0.5, 0.0, 0.0), (-0.5, 0.25, 10.0)]:
        r_repo = reward(a, b, corr=rho, noise=tau, R=1, Z=0, P=p)
        assert np.allclose(U(a, b, p, tau, rho), r_repo, atol=1e-12), (rho, tau, p)
    print("U == game.reward: ok")

    # thresholds (table after eq. 35)
    for p, tc_ref in [(0.0, 0.132116913), (1.0, 0.107294600), (10.0, 0.023485399)]:
        tc = tau_c(p)
        assert abs(tc - tc_ref) < 1e-8, (p, tc, tc_ref)
        print(f"tau_c({p:g}) = {tc:.9f}  (ref {tc_ref})   tau0 = {tau0(p):.9f}")

    # pure frictional phase check: p=1, tau=0.12 (r = 0.1328829857)
    eq = solve_frictional(1.0, 0.12)
    print(f"pure p=1 tau=0.12: r = {eq.atoms[0]:.10f} (ref 0.1328829857), "
          f"u_total = {eq.u_total:.10f} (ref 0.7165761408), gap = {eq.check['br_gap']:.1e}")
    assert abs(eq.atoms[0] - 0.1328829857) < 1e-9
    assert abs(eq.u_total - 0.7165761408) < 1e-9

    # two-atom phase check: p=1, tau=0.1
    eq = solve_frictional(1.0, 0.1)
    x, alpha = eq.atoms[1], eq.weights[0]
    print(f"2-atom p=1 tau=0.1: x = {x:.10f} (ref 0.1878324074), "
          f"alpha = {alpha:.10f} (ref 0.0557644403), rbar = {eq.mean:.10f} "
          f"(ref 0.1773580383), u_total = {eq.u_total:.10f} (ref 0.6138280496)")
    assert abs(x - 0.1878324074) < 1e-9 and abs(alpha - 0.0557644403) < 1e-9
    assert abs(eq.mean - 0.1773580383) < 1e-9
    assert abs(eq.u_total - 0.6138280496) < 1e-9

    # three-atom example: p=10, tau=0.01 (section 5)
    eq = solve_frictional(10.0, 0.01)
    print(f"3-atom p=10 tau=0.01: x = {np.array2string(eq.atoms, precision=10)} "
          f"w = {np.array2string(eq.weights, precision=8)}")
    print(f"  rbar = {eq.mean:.13f} (ref 0.0292270234679), "
          f"u_total = {eq.u_total:.12f} (ref 0.414605311741), gap = {eq.check['br_gap']:.1e}")
    assert np.allclose(eq.atoms, [0.0, 0.0253848276193, 0.0536891318757], atol=1e-8)
    assert np.allclose(eq.weights, [0.31421040, 0.26824276, 0.41754684], atol=1e-7)
    assert abs(eq.mean - 0.0292270234679) < 1e-9
    assert abs(eq.u_total - 0.414605311741) < 1e-9

    # zero-risk phase: p=10, tau=0.1 >= tau0 = 1/42
    eq = solve_frictional(10.0, 0.1)
    assert eq.atoms.tolist() == [0.0] and eq.u_total == 1.0
    print(f"zero-risk p=10 tau=0.1: gap = {eq.check['br_gap']:.1e}")

    # tau=0 closed forms (13)-(18) at p=1 vs the correlation-only check table
    for rho, mu_ref in [(-1.0, 0.25), (0.0, 0.236068), (1.0, 0.213061)]:
        eq = solve_tau0(1.0, rho)
        print(f"tau=0 p=1 rho={rho:+.0f}: mu = {eq.mean:.6f} (ref {mu_ref}), "
              f"indifference resid = {eq.check['indifference']:.1e}")
        assert abs(eq.mean - mu_ref) < 1e-5

    # Fredholm and shooting solvers vs closed form at rho=0
    for p in (0.0, 1.0):
        cf = solve_tau0(p, 0.0)
        fr = solve_tau0(p, 0.0, closed_form=False, method="fredholm")
        sh = solve_tau0(p, 0.0, closed_form=False, method="shooting")
        print(f"tau=0 rho=0 p={p:g}: closed h={cf.h:.10f} mu={cf.mean:.10f} | "
              f"Fredholm h={fr.h:.10f} mu={fr.mean:.10f} | "
              f"shooting h={sh.h:.10f} mu={sh.mean:.10f}")
        assert abs(cf.h - fr.h) < 2e-5 and abs(cf.mean - fr.mean) < 2e-5
        assert abs(cf.h - sh.h) < 1e-7 and abs(cf.mean - sh.mean) < 1e-7

    # shooting vs Fredholm for rho != 0 (Fredholm is only accurate to ~1e-5
    # there: its uniform grid does not resolve the power-law term of f at 0)
    for p, rho in [(1.0, 0.5), (0.0, 0.5), (1.0, -0.5)]:
        fr = solve_tau0(p, rho, method="fredholm")
        sh = solve_tau0(p, rho, method="shooting")
        print(f"tau=0 rho={rho:+g} p={p:g}: Fredholm h={fr.h:.10f} | shooting "
              f"h={sh.h:.10f} mu={sh.mean:.10f}, endpoint identity "
              f"{sh.check['endpoint_identity']:.1e}, indifference {sh.check['indifference']:.1e}")
        assert abs(fr.h - sh.h) < 2e-5
        assert sh.check["endpoint_identity"] < 1e-7

    print("all checks passed")
