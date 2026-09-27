"""
n-player CfR game with market friction, solved with the Multiple Oracle
algorithm of Kroupa & Votroubek, "Multiple Oracle Algorithm to Solve
Continuous Games" (arXiv:2109.04178, Algorithm 1).

Game (independent failures, rho = 0). Player i picks a risk r_i in [0, 1],
fails with probability r_i and then pays P. Among the players that do not
fail, player i wins R = 1 with probability

    exp(r_i / tau) / sum_{j survives} exp(r_j / tau)        (softmax),

which for n = 2 is sigma_tau(r_1 - r_2), the friction model of game.py.

Expected payoff of the risk r when the n - 1 opponents play the mixed
strategy (a_k, w_k): with 1/x = int_0^inf exp(-t x) dt and t = exp(s),

    U(r) = -P r + (1 - r) int exp(s) exp(-exp(s)) G(s, r)^(n-1) ds,
    G(s, r) = sum_k w_k [a_k + (1 - a_k) exp(-exp(s + (a_k - r) / tau))],

one integral over s whatever n (trapezoid rule on s in [-40, 4]).
"""

import time

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize_scalar
from scipy.stats import qmc

S = np.arange(-40.0, 4.0 + 1e-9, 0.1)  # quadrature nodes in s = log t
OMEGA = np.exp(S - np.exp(S)) * 0.1  # weights exp(s) exp(-exp(s)) ds


@njit(inline="always")
def _kernel(a, r, s, tau):
    e = s + (a - r) / tau
    return a + (1.0 - a) * (np.exp(-np.exp(e)) if e < 700.0 else 0.0)


@njit(parallel=True, cache=True)
def payoffs(r, atoms, weights, n, tau, P, s=S, omega=OMEGA):
    """U(r_q) for each q: payoff of r_q against n - 1 opponents playing
    (atoms, weights)."""
    out = np.empty(len(r))
    for q in prange(len(r)):
        acc = 0.0
        for m in range(len(s)):
            g = 0.0
            for k in range(len(atoms)):
                g += weights[k] * _kernel(atoms[k], r[q], s[m], tau)
            acc += omega[m] * g ** (n - 1)
        out[q] = -P * r[q] + (1.0 - r[q]) * acc
    return out


@njit(cache=True)
def payoffs_and_jacobian(atoms, weights, n, tau, P, s=S, omega=OMEGA):
    """Restricted symmetric game on the points `atoms`: U_i = payoff of atom i
    when the opponents play `weights`, and J_ik = dU_i / dweights_k."""
    k = len(atoms)
    U = np.empty(k)
    J = np.zeros((k, k))
    K = np.empty(k)
    for i in range(k):
        acc = 0.0
        for m in range(len(s)):
            g = 0.0
            for j in range(k):
                K[j] = _kernel(atoms[j], atoms[i], s[m], tau)
                g += weights[j] * K[j]
            acc += omega[m] * g ** (n - 1)
            c = omega[m] * (n - 1) * g ** (n - 2)
            for j in range(k):
                J[i, j] += c * K[j]
        U[i] = -P * atoms[i] + (1.0 - atoms[i]) * acc
        for j in range(k):
            J[i, j] *= 1.0 - atoms[i]
    return U, J


# ---------------------------------------------------------------------------
# restricted game: symmetric equilibrium of the finite symmetric n-player game
# ---------------------------------------------------------------------------


def _gap(atoms, x, n, tau, P):
    U = payoffs(atoms, atoms, x, n, tau, P)
    return float(U.max() - x @ U), float(x @ U)


def _support_newton(atoms, x, n, tau, P, thresh=1e-9, iters=30):
    """Newton on the support S of x: U_S(x_S) = v 1, sum(x_S) = 1, x = 0
    outside S. Returns the solution if it is an equilibrium, else None."""
    S = np.nonzero(x > thresh * x.max())[0]
    a, xs = atoms[S], x[S] / x[S].sum()
    v = xs @ payoffs(a, a, xs, n, tau, P)
    k = len(S)
    for _ in range(iters):
        U, J = payoffs_and_jacobian(a, xs, n, tau, P)
        F = np.append(U - v, xs.sum() - 1)
        if np.abs(F).max() < 1e-15:
            break
        M = np.zeros((k + 1, k + 1))
        M[:k, :k] = J
        M[:k, k] = -1
        M[k, :k] = 1
        try:
            d = np.linalg.solve(M, -F)
        except np.linalg.LinAlgError:
            return None
        xs, v = xs + d[:k], v + d[k]
    if xs.min() < 0:
        return None
    q = np.zeros(len(x))
    q[S] = xs
    return q if _gap(atoms, q, n, tau, P)[0] < 1e-12 else None


def restricted_equilibrium(atoms, n, tau, P, p0=None, tol=1e-13, max_newton=300,
                           polish_after=40):
    """Symmetric equilibrium x of the game restricted to `atoms`:
    x >= 0, v - U(x) >= 0, x_i (v - U_i(x)) = 0, sum(x) = 1.

    Smoothed Fischer-Burmeister + Newton (a semismooth method for the
    complementarity problem, the role PATH plays in the paper); after
    `polish_after` steps, and at the end, Newton on the detected support."""
    k = len(atoms)
    x = np.full(k, 1.0 / k) if p0 is None else np.clip(p0, 0, None) / np.clip(p0, 0, None).sum()
    U, J = payoffs_and_jacobian(atoms, x, n, tau, P)
    v = U.max()

    def residual(x, v, mu, jac=True):
        if jac:
            U, J = payoffs_and_jacobian(atoms, x, n, tau, P)
        else:  # line search: payoffs only
            U, J = payoffs(atoms, atoms, x, n, tau, P), None
        b = v - U
        r = np.sqrt(x * x + b * b + 2 * mu * mu)
        return np.append(x + b - r, x.sum() - 1), b, r, J

    steps = 0
    for mu in ([1e-2, 1e-3, 1e-4] if p0 is None else []) + [1e-6, 1e-8, 1e-10, 1e-12, 0.0]:
        for _ in range(60):
            F, b, r, J = residual(x, v, mu)
            norm = np.linalg.norm(F)
            if norm < max(mu, tol) or steps >= max_newton:
                break
            if steps == polish_after:
                q = _support_newton(atoms, np.clip(x, 0, None), n, tau, P)
                if q is not None:
                    g, val = _gap(atoms, q, n, tau, P)
                    return q, val, g, steps
            r = np.maximum(r, 1e-300)
            da, db = 1 - x / r, 1 - b / r
            M = np.empty((k + 1, k + 1))
            M[:k, :k] = -db[:, None] * J
            M[np.arange(k), np.arange(k)] += da
            M[:k, k] = db
            M[k, :k] = 1
            M[k, k] = 0
            try:
                d = np.linalg.solve(M, -F)
            except np.linalg.LinAlgError:
                d = np.linalg.lstsq(M, -F, rcond=None)[0]
            steps += 1
            step = 1.0
            while step > 1e-10:
                Fn = residual(x + step * d[:k], v + step * d[k], mu, jac=False)[0]
                if np.linalg.norm(Fn) <= (1 - 1e-4 * step) * norm:
                    break
                step /= 2
            x, v = x + step * d[:k], v + step * d[k]
    x = np.clip(x, 0, None)
    x /= x.sum()
    q = _support_newton(atoms, x, n, tau, P)
    if q is not None:
        x = q
    g, val = _gap(atoms, x, n, tau, P)
    return x, val, g, steps


# ---------------------------------------------------------------------------
# best response oracle: global optimization of U over [0, 1]
# ---------------------------------------------------------------------------


def best_response(atoms, weights, n, tau, P, log_points=12, refine=4, seed=0):
    """max over r in [0, 1] of U(r): Sobol scan, then Brent refinement."""
    scan = qmc.Sobol(d=1, seed=seed).random_base2(log_points).ravel()
    pts = np.concatenate([scan, [0.0, 1.0], atoms])
    vals = payoffs(pts, atoms, weights, n, tau, P)
    h = 2.0**-log_points
    best_r, best_v = pts[np.argmax(vals)], vals.max()
    for i in np.argsort(vals)[-refine:]:
        lo, hi = max(0.0, pts[i] - 2 * h), min(1.0, pts[i] + 2 * h)
        res = minimize_scalar(
            lambda r: -payoffs(np.array([r]), atoms, weights, n, tau, P)[0],
            bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        if -res.fun > best_v:
            best_r, best_v = float(res.x), float(-res.fun)
    return float(best_v), float(best_r)


# ---------------------------------------------------------------------------
# Multiple Oracle (Kroupa & Votroubek, Algorithm 1), symmetric form
# ---------------------------------------------------------------------------


def multiple_oracle(n, tau, P=1.0, eps=1e-9, max_iter=300, time_limit=600,
                    log_points=12, prune=False, verbose=False):
    """Algorithm 1 of arXiv:2109.04178 for the symmetric n-player CfR game.

    The game is symmetric, so every player has the same strategy set X^j and
    the same best response: X^{j+1} = X^j u {x^{j+1}}. Stop when the sum of
    the players' best-response gains n (U(x^{j+1}, p^j) - U(p^j)) <= eps.

    prune=True (not in the paper): before adding the best response, drop the
    strategies with zero weight in the restricted equilibrium. On this game it
    cycles (n = 2, tau = 0.03: after 100 iterations the set has 2 strategies
    and NashConv repeats 0.56, 0.43, 0.32, 0.24, 0.20); the paper keeps every
    strategy.
    """
    start = time.perf_counter()
    atoms, x = np.array([0.0]), None
    history = []
    for j in range(1, max_iter + 1):
        # master problem: equilibrium of the finite subgame (warm start)
        p0 = None if x is None else np.append(x, 1e-3)[np.argsort(order)]
        x, v, restricted_gap, steps = restricted_equilibrium(atoms, n, tau, P, p0)
        # subproblem: best response oracle over the continuous set [0, 1]
        br_v, br_r = best_response(atoms, x, n, tau, P, log_points=log_points)
        nashconv = n * (br_v - v)
        history.append((j, len(atoms), nashconv, time.perf_counter() - start))
        if verbose:
            print(f"  it {j:3d}  t={time.perf_counter() - start:6.1f}s  |X|={len(atoms):3d}  NashConv={nashconv:.2e}  "
                  f"restricted gap={restricted_gap:.1e}  newton={steps}")
        if nashconv <= eps or time.perf_counter() - start > time_limit:
            break
        if np.min(np.abs(atoms - br_r)) < 1e-12:  # best response already in X
            break
        if prune:
            keep = x > 1e-12
            atoms, x = atoms[keep], x[keep]
        new = np.append(atoms, br_r)
        order = np.argsort(new)
        atoms = new[order]
    return atoms, x, {"iterations": j, "nashconv": nashconv, "value": v,
                      "time": time.perf_counter() - start, "history": history}


def exploitability(atoms, x, n, tau, P, log_points=16):
    """NashConv of the symmetric profile, with a finer best response than the
    oracle (independent check)."""
    v = x @ payoffs(atoms, atoms, x, n, tau, P)
    br_v, _ = best_response(atoms, x, n, tau, P, log_points=log_points, refine=8, seed=1)
    return n * (br_v - v)
