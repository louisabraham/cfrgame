"""
Baseline equilibrium solvers for the 2-player CfR game, compared with our
regret-matching solver in `benchmark.py`.

Grid methods take the payoff matrix `A[i, j] = u(x_i, x_j)` of the symmetric
game on the grid `x` (player 1 plays `A`, player 2 plays `A.T`).
All methods return `(actions1, prob1, actions2, prob2, info)`.
"""

import time

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize_scalar
from scipy.stats import qmc

from game import reward, reward_matrix

# The global optimum of the Nash-gap programs is 0 but the solvers cannot prove
# it (the lower bound stays negative); stop at the first incumbent below this.
GAP_STOP = 1e-7


def _simplex(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
    return p / p.sum()


def grid_gap(A, p1, p2):
    """NashConv of (p1, p2) in the bimatrix game (A, A.T)."""
    out1 = A @ p2
    out2 = A @ p1
    return out1.max() - p1 @ out1 + out2.max() - p2 @ out2


# ---------------------------------------------------------------------------
# continuous best response
# ---------------------------------------------------------------------------


@njit(parallel=True)
def _expected_rewards(points, actions, probs, corr, noise, R, Z, P):
    out = np.empty(len(points))
    for i in prange(len(points)):
        out[i] = reward(points[i], actions, corr, noise, R, Z, P) @ probs
    return out


def expected_rewards(points, actions, probs, game_parameters):
    g = game_parameters
    return _expected_rewards(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(actions, dtype=np.float64),
        np.ascontiguousarray(probs, dtype=np.float64),
        float(g["corr"]), float(g["noise"]), float(g["R"]), float(g["Z"]), float(g["P"]),
    )


def continuous_best_response(game_parameters, actions, probs, log_points=14,
                             refine=8, seed=0, side=1e-12):
    """max over r in [0, 1] of E[u(r, X)], X ~ (actions, probs).

    Scans a scrambled Sobol set (one point in each interval of length
    2**-log_points), then refines the `refine` best points with Brent's method.
    The opponent's atoms +-side are also scanned: at tau = 0 the payoff jumps
    there and the supremum is a one-sided limit.
    """
    keep = probs > 0
    actions, probs = actions[keep], probs[keep]
    sobol = qmc.Sobol(d=1, seed=seed).random_base2(log_points).ravel()
    sides = [] if side is None else np.clip(
        np.concatenate([actions - side, actions + side]), 0.0, 1.0)
    points = np.concatenate([sobol, [0.0, 1.0], actions, sides])
    values = expected_rewards(points, actions, probs, game_parameters)
    h = 2.0**-log_points
    best_r, best_v = points[np.argmax(values)], values.max()
    for i in np.argsort(values)[-refine:]:
        lo, hi = max(0.0, points[i] - 2 * h), min(1.0, points[i] + 2 * h)
        res = minimize_scalar(
            lambda r: -expected_rewards(np.array([r]), actions, probs, game_parameters)[0],
            bounds=(lo, hi), method="bounded", options={"xatol": 1e-12},
        )
        if -res.fun > best_v:
            best_r, best_v = res.x, -res.fun
    return float(best_v), float(best_r)


# ---------------------------------------------------------------------------
# 1. Lemke-Howson (Gambit)
# ---------------------------------------------------------------------------


def gambit_lh(A, x, **_):
    import pygambit as gbt

    t = time.perf_counter()
    game = gbt.Game.from_arrays(A, A.T)
    t_build = time.perf_counter() - t
    res = gbt.nash.lcp_solve(game, rational=False, stop_after=1)
    eq = res.equilibria[0]
    p1, p2 = (
        np.array([float(eq[s]) for s in player.strategies]) for player in game.players
    )
    return x, _simplex(p1), x, _simplex(p2), {"t_build": t_build}


def gambit_logit(A, x, maxregret=1e-8, **_):
    """Gambit: follow the principal branch of the logit quantal response
    equilibrium correspondence (lambda -> infinity) to a Nash equilibrium."""
    import pygambit as gbt

    t = time.perf_counter()
    game = gbt.Game.from_arrays(A, A.T)
    t_build = time.perf_counter() - t
    eq = gbt.nash.logit_solve(game, maxregret=maxregret).equilibria[0]
    p1, p2 = (
        np.array([float(eq[s]) for s in player.strategies]) for player in game.players
    )
    return x, _simplex(p1), x, _simplex(p2), {"t_build": t_build}


def _pivot(T, basis, col, slack, tol=1e-12):
    """Minimum-ratio pivot on column `col`; returns the label that leaves.

    Ties are broken by the lexicographic rule on the columns `slack` (the
    initial identity basis), which prevents cycling in degenerate games.
    """
    c = T[:, col]
    rows = np.nonzero(c > tol)[0]
    if len(rows) == 0:
        raise RuntimeError("ray termination")
    for k in [-1, *slack]:
        ratios = T[rows, k] / c[rows]
        lo = ratios.min()
        rows = rows[ratios <= lo + 1e-12 * abs(lo) + 1e-15]  # relative ties only
        if len(rows) == 1:
            break
    r = rows[0]
    T[r] /= T[r, col]
    f = T[:, col].copy()
    f[r] = 0
    T -= np.outer(f, T[r])
    leaving, basis[r] = basis[r], col
    return leaving


def lemke_howson(A, B, dropped=0, max_pivots=100000):
    """Lemke-Howson on the bimatrix game (A, B), tableau form as in
    https://math-econ-code.github.io/gt_nash.html

    With positive payoffs, x >= 0, y >= 0, s = 1 - Ay >= 0, t = 1 - B'x >= 0,
    x's = y't = 0. Labels: x_i and s_i share label i, y_j and t_j label n + j.
    """
    A = A - A.min() + 1
    B = B - B.min() + 1
    n, m = A.shape
    # rows t_j: B'x + t = 1 (columns x_0..x_{n-1}, t_0..t_{m-1})
    T2 = np.hstack([B.T, np.eye(m), np.ones((m, 1))])
    basis2 = list(range(n, n + m))
    # rows s_i: s + Ay = 1 (columns s_0..s_{n-1}, y_0..y_{m-1})
    T1 = np.hstack([np.eye(n), A, np.ones((n, 1))])
    basis1 = list(range(n))
    # the column of a variable is its label in both tableaux; the complement
    # of the leaving variable enters the other tableau
    tab2 = (T2, basis2, range(n, n + m))  # identity on the t columns
    tab1 = (T1, basis1, range(n))  # identity on the s columns
    tableaux = [tab2, tab1] if dropped < n else [tab1, tab2]
    entering = dropped
    for pivots in range(1, max_pivots + 1):
        T, basis, slack = tableaux[(pivots - 1) % 2]
        entering = _pivot(T, basis, entering, slack)
        if entering == dropped:
            break
    x, y = np.zeros(n), np.zeros(m)
    for r, label in enumerate(basis2):
        if label < n:
            x[label] = T2[r, -1]
    for r, label in enumerate(basis1):
        if label >= n:
            y[label - n] = T1[r, -1]
    return _simplex(x), _simplex(y), pivots


def lh_tableau(A, x, **_):
    p1, p2, pivots = lemke_howson(A, A.T)
    return x, p1, x, p2, {"pivots": pivots}


# ---------------------------------------------------------------------------
# 2. Bilinear Nash-gap program: min u + v - x'(A + B)y
#    With w = A y and s = B'x = A x the objective has only 2n bilinear terms:
#    min u + v - x'w - y's  s.t.  w <= u, s <= v, x, y in simplex.
# ---------------------------------------------------------------------------


def gurobi_qp(A, x, time_limit=300, **_):
    import gurobipy as gp

    n = len(x)
    with gp.Env(params={"OutputFlag": 0, "Threads": 1}) as env, gp.Model(env=env) as m:
        m.Params.TimeLimit = time_limit
        m.Params.NonConvex = 2
        m.Params.FeasibilityTol = 1e-9
        m.Params.BestObjStop = GAP_STOP
        p1 = m.addMVar(n, lb=0, ub=1)
        p2 = m.addMVar(n, lb=0, ub=1)
        w = m.addMVar(n, lb=-gp.GRB.INFINITY)
        s = m.addMVar(n, lb=-gp.GRB.INFINITY)
        u = m.addVar(lb=-gp.GRB.INFINITY)
        v = m.addVar(lb=-gp.GRB.INFINITY)
        m.addConstr(w == A @ p2)
        m.addConstr(s == A @ p1)
        m.addConstr(w <= u)
        m.addConstr(s <= v)
        m.addConstr(p1.sum() == 1)
        m.addConstr(p2.sum() == 1)
        m.setObjective(u + v - p1 @ w - p2 @ s, gp.GRB.MINIMIZE)
        m.optimize()
        if m.SolCount == 0:
            raise RuntimeError(f"no solution, status {m.Status}")
        info = {"status": int(m.Status), "obj": m.ObjVal, "bound": m.ObjBound}
        return x, _simplex(p1.X), x, _simplex(p2.X), info


def mangasarian_stone(A, x, time_limit=300, **_):
    """Mangasarian-Stone QP with the dense objective alpha + beta - p'(A+B)q."""
    import gurobipy as gp

    n = len(x)
    with gp.Env(params={"OutputFlag": 0, "Threads": 1}) as env, gp.Model(env=env) as m:
        m.Params.TimeLimit = time_limit
        m.Params.NonConvex = 2
        m.Params.FeasibilityTol = 1e-9
        m.Params.BestObjStop = GAP_STOP
        p = m.addMVar(n, lb=0)
        q = m.addMVar(n, lb=0)
        alpha = m.addMVar(1, lb=-gp.GRB.INFINITY)
        beta = m.addMVar(1, lb=-gp.GRB.INFINITY)
        ones = np.ones((n, 1))
        m.addConstr(ones @ alpha - A @ q >= 0)
        m.addConstr(ones @ beta - A @ p >= 0)  # B'p with B = A'
        m.addConstr(p.sum() == 1)
        m.addConstr(q.sum() == 1)
        m.setObjective(alpha.sum() + beta.sum() - p @ (A + A.T) @ q, gp.GRB.MINIMIZE)
        m.optimize()
        if m.SolCount == 0:
            raise RuntimeError(f"no solution, status {m.Status}")
        info = {"status": int(m.Status), "obj": m.ObjVal, "bound": m.ObjBound}
        return x, _simplex(p.X), x, _simplex(q.X), info


def scip_qp(A, x, time_limit=300, **_):
    from pyscipopt import Model, quicksum

    n = len(x)
    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/primal", GAP_STOP)
    m.setParam("numerics/feastol", 1e-9)
    p1 = [m.addVar(lb=0, ub=1) for _ in range(n)]
    p2 = [m.addVar(lb=0, ub=1) for _ in range(n)]
    w = [m.addVar(lb=None) for _ in range(n)]
    s = [m.addVar(lb=None) for _ in range(n)]
    u, v, z = m.addVar(lb=None), m.addVar(lb=None), m.addVar(lb=None)
    for i in range(n):
        m.addCons(w[i] == quicksum(A[i, j] * p2[j] for j in range(n)))
        m.addCons(s[i] == quicksum(A[i, j] * p1[j] for j in range(n)))
        m.addCons(w[i] <= u)
        m.addCons(s[i] <= v)
    m.addCons(quicksum(p1) == 1)
    m.addCons(quicksum(p2) == 1)
    # SCIP needs a linear objective: z >= -(x'w + y's)
    m.addCons(z + quicksum(p1[i] * w[i] + p2[i] * s[i] for i in range(n)) >= 0)
    m.setObjective(u + v + z, "minimize")
    m.optimize()
    if m.getNSols() == 0:
        raise RuntimeError(f"no solution, status {m.getStatus()}")
    sol = m.getBestSol()
    info = {"status": m.getStatus(), "obj": m.getObjVal(), "bound": m.getDualbound()}
    return (
        x, _simplex([sol[a] for a in p1]), x, _simplex([sol[a] for a in p2]), info,
    )


# ---------------------------------------------------------------------------
# 3. Symmetric NCP with Fischer-Burmeister and smoothing Newton
#    x_i >= 0, v - (Ax)_i >= 0, x_i (v - (Ax)_i) = 0, sum(x) = 1
# ---------------------------------------------------------------------------


def _fb(x, v, A, mu):
    b = v - A @ x
    r = np.sqrt(x * x + b * b + 2 * mu * mu)
    F = np.append(x + b - r, x.sum() - 1)
    return F, b, r


def fb_ncp(A, x, tol=1e-13, time_limit=300, p0=None, **_):
    """p0: warm start (then the smoothing starts at a small mu)."""
    n = len(x)
    start = time.perf_counter()
    p = np.full(n, 1.0 / n) if p0 is None else _simplex(p0)
    v = (A @ p).max()
    newton = 0
    mus = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 0.0]
    for mu in mus if p0 is None else mus[4:]:
        for _ in range(100):
            F, b, r = _fb(p, v, A, mu)
            norm = np.linalg.norm(F)
            if norm < max(mu, tol) or time.perf_counter() - start > time_limit:
                break
            r = np.maximum(r, 1e-300)
            da, db = 1 - p / r, 1 - b / r
            J = np.empty((n + 1, n + 1))
            J[:n, :n] = -db[:, None] * A
            J[np.arange(n), np.arange(n)] += da
            J[:n, n] = db
            J[n, :n] = 1
            J[n, n] = 0
            try:
                d = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                d = np.linalg.lstsq(J, -F, rcond=None)[0]
            newton += 1
            step = 1.0
            while step > 1e-10:
                Fn = _fb(p + step * d[:n], v + step * d[n], A, mu)[0]
                if np.linalg.norm(Fn) <= (1 - 1e-4 * step) * norm:
                    break
                step /= 2
            p, v = p + step * d[:n], v + step * d[n]
    F = _fb(p, v, A, 0.0)[0]
    p = _simplex(p)
    return x, p, x, p, {"newton": newton, "residual": float(np.linalg.norm(F))}


# ---------------------------------------------------------------------------
# 4. Double Oracle with (continuous or grid) best responses
# ---------------------------------------------------------------------------


def _support_polish(M, p, tol=1e-9):
    """Exact symmetric equilibrium on the support of p: solve M_SS q = v 1,
    sum(q) = 1. Returns None if q is not an equilibrium of M."""
    for thresh in (tol, 1e-6, 1e-4):
        S = np.nonzero(p > thresh * p.max())[0]
        k = len(S)
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = M[np.ix_(S, S)]
        K[:k, k] = -1
        K[k, :k] = 1
        rhs = np.append(np.zeros(k), 1.0)
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            continue
        if sol[:k].min() < 0:
            continue
        q = np.zeros(len(p))
        q[S] = sol[:k]
        if grid_gap(M, q, q) < 1e-12:
            return q
    return None


def _restricted_solve(M, p0=None, max_pivots=20000):
    """Symmetric equilibrium of a restricted symmetric game.

    Fischer-Burmeister Newton (warm-started from p0, then cold), each followed
    by an exact solve on the detected support; Lemke-Howson as a last resort.
    Returns the candidate with the smallest gap.
    """
    idx = np.arange(len(M))
    cands = []
    for start in ([p0, None] if p0 is not None else [None]):
        _, p, _, _, _ = fb_ncp(M, idx, time_limit=30, p0=start)
        q = _support_polish(M, p)
        for c in (p, q):
            if c is not None:
                cands.append((grid_gap(M, c, c), c, c))
        if min(c[0] for c in cands) < 1e-12:
            break
    if min(c[0] for c in cands) > 1e-10:
        p1, p2, _ = lemke_howson(M, M.T, max_pivots=max_pivots)
        cands.append((grid_gap(M, p1, p2), p1, p2))
    _, p1, p2 = min(cands, key=lambda c: c[0])
    return p1, p2


def _top_local_maxima(points, values, above, m):
    """Up to m local maxima of values (points sorted) with value > above."""
    order = np.argsort(points)
    x, y = points[order], values[order]
    left = np.concatenate([[-np.inf], y[:-1]])
    right = np.concatenate([y[1:], [-np.inf]])
    peaks = np.nonzero((y >= left) & (y >= right) & (y > above))[0]
    peaks = peaks[np.argsort(-y[peaks])][:m]
    return list(x[peaks])


def double_oracle(game_parameters, grid=None, tol=1e-10, max_iter=500,
                  time_limit=300, log_points=12, oracles=8, **_):
    """Support generation on a restricted game.

    grid=None: best responses over r in [0, 1] (no payoff matrix at all),
        scan of 2**log_points Sobol points refined with Brent's method.
    grid=array: best responses over the grid points (exact grid equilibrium);
        the strategies are returned on the full grid.
    Each iteration adds, for each player, the best response and up to
    `oracles - 1` other profitable local maxima of the deviation payoff.
    """
    start = time.perf_counter()
    jump = game_parameters["noise"] <= 0  # payoff jumps at r1 = r2
    side = 2.0**-log_points
    if grid is None:
        scan = qmc.Sobol(d=1, seed=0).random_base2(log_points).ravel()
    support = np.array([0.0])
    it = 0
    for it in range(1, max_iter + 1):
        M = reward_matrix(support, support, **game_parameters)
        if it > 1:  # warm start: previous equilibrium, small mass on new points
            prev = np.interp(support, old_support, p_sym, left=0, right=0)
            prev[~np.isin(support, old_support)] = 1e-3 / len(support)
        p1, p2 = _restricted_solve(M, prev if it > 1 else None)
        old_support, p_sym = support, (p1 + p2) / 2
        v1, v2 = p1 @ M @ p2, p2 @ M @ p1
        new = []
        gap = 0.0
        for opp, v in ((p2, v1), (p1, v2)):
            if grid is None:
                br_v, br_r = continuous_best_response(
                    game_parameters, support, opp, log_points=log_points, refine=4,
                    side=side if jump else None,
                )
                pts = scan if not jump else np.concatenate(
                    [scan, np.clip(np.concatenate([support - side, support + side]), 0, 1)])
                cands = [br_r] + _top_local_maxima(
                    pts, expected_rewards(pts, support, opp, game_parameters),
                    v + tol, oracles - 1)
            else:
                values = expected_rewards(grid, support, opp, game_parameters)
                br_v = values.max()
                cands = _top_local_maxima(grid, values, v + tol, oracles)
            gap += br_v - v
            for r in cands:
                if grid is None and jump:
                    # at tau = 0 the supremum is a one-sided limit at a support
                    # point: take the point at distance `side` on that side
                    j = np.argmin(np.abs(support - r))
                    if abs(support[j] - r) < side:
                        r = float(np.clip(
                            support[j] + (side if r >= support[j] else -side), 0.0, 1.0))
                taken = np.concatenate([support, new])
                if np.min(np.abs(taken - r)) > 1e-12:
                    new.append(r)
        if gap < tol or not new or time.perf_counter() - start > time_limit:
            break
        support = np.unique(np.concatenate([support, new]))
    # return the support of the last restricted solve (the loop can end at
    # max_iter just after adding new points)
    support = old_support
    info = {"iterations": it, "support": len(support), "restricted_gap": float(gap)}
    if grid is None:
        return support, p1, support, p2, info
    idx = np.searchsorted(grid, support)
    q1, q2 = np.zeros(len(grid)), np.zeros(len(grid))
    q1[idx], q2[idx] = p1, p2
    return grid, q1, grid, q2, info


# ---------------------------------------------------------------------------
# 5. Ipopt, multi-start, on the symmetric Nash-gap program
#    min v - x'Ax  s.t.  Ax <= v, sum(x) = 1, x >= 0
# ---------------------------------------------------------------------------


class _IpoptGap:
    def __init__(self, A):
        self.A = A
        self.S = A + A.T
        self.n = len(A)
        self.tril = np.tril_indices(self.n)

    def objective(self, z):
        p = z[:-1]
        return z[-1] - p @ self.A @ p

    def gradient(self, z):
        return np.append(-self.S @ z[:-1], 1.0)

    def constraints(self, z):
        return np.append(self.A @ z[:-1] - z[-1], z[:-1].sum())

    def jacobian(self, z):
        n = self.n
        J = np.zeros((n + 1, n + 1))
        J[:n, :n] = self.A
        J[:n, n] = -1
        J[n, :n] = 1
        return J.ravel()

    def hessianstructure(self):
        return self.tril

    def hessian(self, z, lagrange, obj_factor):
        return -obj_factor * self.S[self.tril]


def ipopt_ms(A, x, starts=5, time_limit=300, seed=0, **_):
    import cyipopt

    n = len(x)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    best = None
    runs = 0
    for k in range(starts):
        remaining = time_limit - (time.perf_counter() - start)
        if remaining <= 1:
            break
        p0 = np.full(n, 1.0 / n) if k == 0 else rng.dirichlet(np.ones(n))
        z0 = np.append(p0, (A @ p0).max())
        prob = cyipopt.Problem(
            n=n + 1, m=n + 1, problem_obj=_IpoptGap(A),
            lb=np.append(np.zeros(n), -1e20), ub=np.append(np.ones(n), 1e20),
            cl=np.append(np.full(n, -1e20), 1.0), cu=np.append(np.zeros(n), 1.0),
        )
        prob.add_option("print_level", 0)
        prob.add_option("sb", "yes")
        prob.add_option("tol", 1e-12)
        prob.add_option("max_iter", 3000)
        prob.add_option("max_wall_time", float(remaining / (starts - k)))
        z, _ = prob.solve(z0)
        runs += 1
        p = _simplex(z[:-1])
        gap = grid_gap(A, p, p)
        if best is None or gap < best[0]:
            best = (gap, p)
        if gap < 1e-10:
            break
    p = best[1]
    return x, p, x, p, {"starts": runs}


# ---------------------------------------------------------------------------
# 6. Learning dynamics
# ---------------------------------------------------------------------------


@njit
def _fictitious_play(A, Ay, Ax, c1, c2, iters):
    for _ in range(iters):
        b1 = np.argmax(Ay)
        b2 = np.argmax(Ax)
        c1[b1] += 1
        c2[b2] += 1
        Ay += A[:, b2]
        Ax += A[:, b1]


def anytime(step, snapshot, time_limit, first=1):
    """Run step(k) in chunks of doubling size and record snapshot() after each.

    Returns a list of (iterations, solve time, snapshot). The time excludes the
    snapshots. The run stops before a chunk would take the total past
    time_limit (a chunk is as long as all the previous ones together).
    """
    out, done, elapsed, k = [], 0, 0.0, first
    while True:
        t = time.perf_counter()
        step(k)
        elapsed += time.perf_counter() - t
        done += k
        out.append((done, elapsed, snapshot()))
        if 2 * elapsed > time_limit:
            return out
        k = done


# One "round" of a sampled method is n updates, so that a round of every
# method below costs O(n^2), like one iteration of CFR.


def regret_matching_anytime(A1, A2, a1, a2, cfr=False, time_limit=60, seed=0):
    """Our solver (regret_matching.py kernels) with snapshots of the average
    strategies. RM: rounds of n sampled updates; CFR: iterations."""
    from regret_matching import _cfr, _rm, _seed_numba, normalize

    _seed_numba(seed)
    n = len(a1)
    R1, R2 = (np.asfortranarray(A1), np.asfortranarray(A2)) if not cfr else (A1, A2)
    reg1, reg2 = np.zeros(n), np.zeros(len(a2))
    avg1, avg2 = np.zeros(n), np.zeros(len(a2))
    kernel = _cfr if cfr else _rm
    per_round = 1 if cfr else n

    def step(k):
        kernel(R1, R2, reg1, reg2, avg1, avg2, k * per_round, False)

    return anytime(step, lambda: (normalize(avg1), normalize(avg2)), time_limit)


@njit
def _fictitious_play(A, Ay, Ax, c1, c2, iters):
    for _ in range(iters):
        b1 = np.argmax(Ay)
        b2 = np.argmax(Ax)
        c1[b1] += 1
        c2[b2] += 1
        Ay += A[:, b2]
        Ax += A[:, b1]


def fictitious_play(A, x, time_limit=60, **_):
    """Simultaneous fictitious play; rounds of n best-response steps."""
    n = len(x)
    A = np.asfortranarray(A)
    Ay, Ax, c1, c2 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    return anytime(lambda k: _fictitious_play(A, Ay, Ax, c1, c2, k * n),
                   lambda: (_simplex(c1), _simplex(c2)), time_limit)


def mwu(A, x, time_limit=60, **_):
    """Hedge (multiplicative weights), simultaneous, average strategies, with
    the anytime step size eta_t = sqrt(8 log n / t)."""
    n = len(x)
    L1, L2 = np.zeros(n), np.zeros(n)
    s1, s2 = np.zeros(n), np.zeros(n)
    t = 0

    def step(k):
        nonlocal t
        for _ in range(k):
            t += 1
            eta = np.sqrt(8 * np.log(n) / t)
            q1 = np.exp(eta * (L1 - L1.max()))
            q1 /= q1.sum()
            q2 = np.exp(eta * (L2 - L2.max()))
            q2 /= q2.sum()
            np.add(s1, q1, out=s1)
            np.add(s2, q2, out=s2)
            np.add(L1, A @ q2, out=L1)
            np.add(L2, A @ q1, out=L2)

    return anytime(step, lambda: (_simplex(s1), _simplex(s2)), time_limit)


def replicator(A, x, time_limit=60, eta=0.25, **_):
    """Discrete replicator dynamics on the symmetric game, last iterate."""
    p = np.full(len(x), 1.0 / len(x))

    def step(k):
        nonlocal p
        for _ in range(k):
            f = A @ p
            p = p * (1 + eta * (f - p @ f))
            p /= p.sum()

    return anytime(step, lambda: (p.copy(), p.copy()), time_limit)


# ---------------------------------------------------------------------------
# nashopt (https://github.com/bemporad/nashopt): the bimatrix game as a
# linear-quadratic GNEP, agent i minimizes -x_i' A x_-i on its simplex.
# ---------------------------------------------------------------------------


def _nashopt_lq(A, solver):
    from nashopt import GNEP_LQ

    n = len(A)
    Z = np.zeros((n, n))
    Q1 = -np.block([[Z, A], [A.T, Z]])
    Q2 = -np.block([[Z, A.T], [A, Z]])
    Aeq = np.zeros((2, 2 * n))
    Aeq[0, :n] = 1
    Aeq[1, n:] = 1
    return GNEP_LQ(
        [n, n], [Q1, Q2], [np.zeros(2 * n)] * 2, lb=np.zeros(2 * n), Aeq=Aeq,
        beq=np.ones(2),
        M=10.0,  # bounds the bound multipliers v - (Ay)_i, at most max(A) - min(A)
        variational=solver not in ("highs", "gurobi"), solver=solver,
    )


def _split(x, z, info=None):
    n = len(x)
    z = np.asarray(z)
    return x, _simplex(z[:n]), x, _simplex(z[n:]), info or {}


def nashopt_milp(A, x, solver="highs", time_limit=300, **_):
    g = _nashopt_lq(A, solver)
    if solver == "gurobi":
        g.mip.model.setParam("Threads", 1)
    else:
        g.mip.setOptionValue("threads", 1)
    sol = g.solve(solver_options={"time_limit": time_limit})
    if sol is None:
        raise RuntimeError(f"no solution within the time limit ({time_limit} s)")
    return _split(x, sol.x)


def nashopt_lemke(A, x, **_):
    return _split(x, _nashopt_lq(A, "lemke").solve().x)


def nashopt_drdaqp(A, x, **_):
    return _split(x, _nashopt_lq(A, "dr_daqp").solve().x)


def nashopt_extragrad(A, x, time_limit=60, **_):
    """Korpelevich extragradient (nashopt has no snapshots: one solve per
    iteration count, 100 * 4^j, each timed on its own)."""
    n = len(x)
    g = _nashopt_lq(A, "extragradient")
    out, maxiter = [], 100
    while True:
        t = time.perf_counter()
        z = np.asarray(g.solve(solver_options={"tol": 1e-14, "stopping": "step",
                                               "maxiter": maxiter}).x)
        elapsed = time.perf_counter() - t
        out.append((maxiter, elapsed, (_simplex(z[:n]), _simplex(z[n:]))))
        if 4 * elapsed > time_limit:
            return out
        maxiter *= 4


def nashopt_gnep(A, x, solver="trf", **_):
    """Nonlinear GNEP: Fischer-Burmeister KKT residual by least squares (JAX)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from nashopt import GNEP

    n = len(x)
    Aj = jnp.array(A)
    f1 = jax.jit(lambda z: -z[:n] @ Aj @ z[n:])
    f2 = jax.jit(lambda z: -z[n:] @ Aj @ z[:n])
    Aeq = np.zeros((2, 2 * n))
    Aeq[0, :n] = 1
    Aeq[1, n:] = 1
    g = GNEP([n, n], f=[f1, f2], lb=np.zeros(2 * n), ub=np.ones(2 * n), Aeq=Aeq,
             beq=np.ones(2), variational=True)
    sol = g.solve(np.full(2 * n, 1.0 / n), solver=solver, verbose=0, max_nfev=500)
    return _split(x, sol.x)
