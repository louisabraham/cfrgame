"""
Hybrid of Double Oracle (baselines.double_oracle) and the moving-atom system
of analytical.py, for the 2-player CfR game with friction (tau > 0).

Each iteration:
1. restricted game on the point set X (weights only, as Double Oracle);
2. merge support points closer than d into one atom (weighted centre),
   for d in `merges` until the move of step 3 succeeds;
3. move the atoms: solve analytical.py's system for positions, weights and
   value, U(x_i) = v for every atom, U_a(x_i) = 0 for interior atoms,
   sum(w) = 1 (atoms at 0 stay at 0); keep the result only if it converges
   with positive weights and no worse best-response gap;
4. continuous best response; stop when its gain is below `tol`;
5. X <- moved atoms + the zero-weight points of X + best response
   (no pruning: dropping points made the Multiple Oracle cycle).
"""

import time

import numpy as np
from scipy.optimize import root

from analytical import U, U_a, q_rho
from baselines import _restricted_solve, continuous_best_response, expected_rewards
from game import reward_matrix


def move_atoms(x, w, p, tau, rho):
    """Newton (hybr) on positions, weights and value; atoms at 0 are fixed.
    Returns (x, w, v) or None if the solve fails."""
    fixed = x < 1e-12
    m, free = len(x), np.nonzero(~fixed)[0]
    A0 = U(x[:, None], x[None, :], p, tau, rho)
    v0 = w @ A0 @ w

    def unpack(z):
        xx = x.copy()
        xx[free] = z[: len(free)]
        return xx, z[len(free): len(free) + m], z[-1]

    def eqs(z):
        xx, ww, vv = unpack(z)
        A = U(xx[:, None], xx[None, :], p, tau, rho)
        B = U_a(xx[:, None], xx[None, :], p, tau, rho)
        return np.concatenate([A @ ww - vv, (B @ ww)[free], [ww.sum() - 1.0]])

    sol = root(eqs, np.concatenate([x[free], w, [v0]]), method="hybr", tol=1e-14)
    xx, ww, vv = unpack(sol.x)
    order = np.argsort(xx)
    xx, ww = xx[order], ww[order]
    # judge by the residual: with tol=1e-14 hybr often reports "xtol too
    # small" (success=False) after it has converged
    ok = (ww.min() > 0 and xx.min() >= 0 and xx.max() <= 1
          and (len(xx) < 2 or np.diff(xx).min() > 1e-9)
          and np.abs(eqs(sol.x)).max() < 1e-10)
    return (xx, ww, float(vv)) if ok else None


def merge_close(x, w, merge):
    """Group sorted support points closer than `merge` into one atom."""
    order = np.argsort(x)
    x, w = x[order], w[order]
    groups = np.split(np.arange(len(x)), np.nonzero(np.diff(x) > merge)[0] + 1)
    wx = np.array([w[g].sum() for g in groups])
    xs = np.array([(x[g] * w[g]).sum() / w[g].sum() for g in groups])
    xs[[x[g].min() < 1e-12 for g in groups]] = 0.0  # keep an atom at 0 at 0
    return xs, wx


def hybrid(p=1.0, tau=0.03, rho=0.5, tol=1e-10, merges=(1e-3, 3e-3, 1e-2, 3e-2),
           max_iter=500, time_limit=300, log_points=12, verbose=False):
    gp = dict(corr=rho, noise=tau, R=1, Z=0, P=p)
    start = time.perf_counter()
    X = np.array([0.0])
    moves = 0
    for it in range(1, max_iter + 1):
        # 1. restricted game (weights only)
        M = reward_matrix(X, X, **gp)
        w1, w2 = _restricted_solve(M)
        wX = (w1 + w2) / 2
        pos = wX > 1e-12
        # 2-3. merge close support points, then move the atoms
        v_rest = wX @ M @ wX
        br_rest, _ = continuous_best_response(gp, X, wX, log_points=log_points, refine=4)
        gap = br_rest - v_rest
        moved = None
        for d in merges:  # first merge distance whose move lowers the gap
            x, w = merge_close(X[pos], wX[pos], d)
            cand = move_atoms(x, w, p, tau, rho)
            if cand is None:
                continue
            xm, wm, vm = cand
            br_m, r_m = continuous_best_response(gp, xm, wm, log_points=log_points, refine=4)
            if br_m - vm <= gap:
                moved = cand
                x, w, gap, moves, br_r = xm, wm, br_m - vm, moves + 1, r_m
                break
        if moved is None:
            x, w = X[pos], wX[pos]
            br_r = continuous_best_response(gp, X, wX, log_points=log_points, refine=4)[1]
        if verbose:
            print(f"  it {it:3d}  t={time.perf_counter() - start:6.1f}s  |X|={len(X):3d}  "
                  f"atoms={len(x):3d}  moved={moved is not None}  gap={gap:.1e}")
        if gap < tol or time.perf_counter() - start > time_limit:
            break
        # 5. new point set: (moved) support + zero-weight points + best response
        rest = X[~pos]
        X = np.unique(np.concatenate([x, rest, [br_r]]))
    return x, w, {"iterations": it, "gap": float(gap), "moves": moves,
                  "time": time.perf_counter() - start, "X": len(X)}


def quasinashconv(p, tau, rho, x, w):
    """Independent check, as benchmark.quasinashconv (finer scan, other seed)."""
    gp = dict(corr=rho, noise=tau, R=1, Z=0, P=p)
    v = w @ expected_rewards(x, x, w, gp)
    br, _ = continuous_best_response(gp, x, w, log_points=14, refine=8, seed=1)
    return 2 * (br - v)


def polish(x, w, p=1.0, tau=0.03, rho=0.5, merges=np.geomspace(1e-6, 5e-2, 40)):
    """Refine a finite-support solution (e.g. the Double Oracle output): for
    each distinct grouping of its support by merge distance d, solve the
    moving-atom system and keep the result with the smallest continuous
    best-response gap. Returns (x, w, info); the input if nothing improves."""
    gp = dict(corr=rho, noise=tau, R=1, Z=0, P=p)
    keep = w > 1e-12
    x, w = x[keep], w[keep] / w[keep].sum()

    def gap(xx, ww):
        v = ww @ expected_rewards(xx, xx, ww, gp)
        return continuous_best_response(gp, xx, ww, log_points=14, refine=8, seed=1)[0] - v

    best = (gap(x, w), x, w, None)
    tried = []
    for d in merges:
        xs, ws = merge_close(x, w, d)
        if any(len(xs) == len(t) and np.allclose(xs, t) for t in tried):
            continue
        tried.append(xs)
        cand = move_atoms(xs, ws, p, tau, rho)
        if cand is None:
            continue
        g = gap(cand[0], cand[1])
        if g < best[0]:
            best = (g, cand[0], cand[1], len(xs))
    g, xb, wb, k = best
    return xb, wb, {"gap": float(g), "atoms": len(xb), "groupings_tried": len(tried),
                    "moved_from": k}


def _system(x, w, v, fixed, p, tau, rho):
    A = U(x[:, None], x[None, :], p, tau, rho)
    B = U_a(x[:, None], x[None, :], p, tau, rho)
    return np.concatenate([A @ w - v, (B @ w)[~fixed], [w.sum() - 1.0]])


def U_b(a, b, p, tau, rho):
    """dU(a, b)/db, from U(a,b) + U(b,a) = 1 - C(a,b) - p (a + b) (identity (2)
    of analytical.py) and the symmetry of C: -dC/db(a,b) - p - U_a(b, a)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return -q_rho(b, a, rho) - p - U_a(b, a, p, tau, rho)


def _jacobian(x, w, fixed, p, tau, rho, h=1e-6):
    """Jacobian of _system in (free positions, weights, v). Weights and v enter
    linearly; for positions, dU/da = U_a, dU/db = U_b, and the second
    derivatives of U_a by central differences (4 matrix evaluations)."""
    m, free = len(x), np.nonzero(~fixed)[0]
    X, Y = x[:, None], x[None, :]
    A = U(X, Y, p, tau, rho)
    B = U_a(X, Y, p, tau, rho)
    Ub = U_b(X, Y, p, tau, rho)
    Baa = (U_a(X + h, Y, p, tau, rho) - U_a(X - h, Y, p, tau, rho)) / (2 * h)
    Bab = (U_a(X, Y + h, p, tau, rho) - U_a(X, Y - h, p, tau, rho)) / (2 * h)
    nf = len(free)
    J = np.zeros((m + nf + 1, nf + m + 1))
    # rows E1_i = sum_j U(x_i, x_j) w_j - v
    dE1 = Ub * w[None, :]
    dE1[np.arange(m), np.arange(m)] += B @ w
    J[:m, :nf] = dE1[:, free]
    J[:m, nf:nf + m] = A
    J[:m, -1] = -1.0
    # rows E2_i = sum_j U_a(x_i, x_j) w_j for interior atoms
    dE2 = Bab * w[None, :]
    dE2[np.arange(m), np.arange(m)] += Baa @ w
    J[m:m + nf, :nf] = dE2[np.ix_(free, free)]
    J[m:m + nf, nf:nf + m] = B[free]
    # row sum(w) = 1
    J[-1, nf:nf + m] = 1.0
    return J


def _merge_pair(x, w, i):
    wi = w[i] + w[i + 1]
    xi = 0.0 if x[i] < 1e-12 else (x[i] * w[i] + x[i + 1] * w[i + 1]) / wi
    return (np.concatenate([x[:i], [xi], x[i + 2:]]),
            np.concatenate([w[:i], [wi], w[i + 2:]]))


def newton_atoms(x, w, p=1.0, tau=0.03, rho=0.5, merge_tol=None, tol=1e-13,
                 max_iter=2000, stall_window=20, stall_ratio=0.9, time_limit=600,
                 max_checks=200):
    """Newton on atom positions, weights and value that handles pairs itself
    (active set):
    - atoms closer than merge_tol (default tau / 10, the length scale of the
      payoff) are merged into one (weighted centre, summed weight);
    - atoms whose weight becomes <= 0 are dropped;
    - if the residual falls by less than 1 - stall_ratio over stall_window
      iterations, the closest pair is merged whatever its distance (a pair
      spaced more than merge_tol still stalls Newton);
    - each step is damped (Levenberg-Marquardt), with the exact Jacobian.
    A forced merge can also join two true atoms, so the start and the state
    before each forced merge are kept, and the state with the smallest
    continuous best-response gap is returned (never worse than the start).
    Atoms at 0 stay at 0.
    Returns (x, w, v, info)."""
    start = time.perf_counter()
    merge_tol = tau / 10 if merge_tol is None else merge_tol
    keep = w > 1e-12
    x, w = x[keep], w[keep] / w[keep].sum()
    v = w @ U(x[:, None], x[None, :], p, tau, rho) @ w
    lam, merges, forced, drops = 1e-6, 0, 0, 0
    history, saved = [], [(x.copy(), w.copy(), float(v))]  # the start is a candidate
    it = 0
    for it in range(max_iter):
        # active set: merge collisions, drop non-positive weights
        order = np.argsort(x)
        x, w = x[order], w[order]
        while len(x) > 1 and np.diff(x).min() < merge_tol:
            x, w = _merge_pair(x, w, int(np.argmin(np.diff(x))))
            merges += 1
        if w.min() <= 0:
            keep = w > 0
            x, w = x[keep], w[keep] / w[keep].sum()
            drops += 1
        fixed = x < 1e-12
        z = np.concatenate([x[~fixed], w, [v]])
        nf = int((~fixed).sum())

        def F(z):
            xx = x.copy()
            xx[~fixed] = z[:nf]
            return _system(xx, z[nf:-1], z[-1], fixed, p, tau, rho)

        f = F(z)
        res = np.abs(f).max()
        if res < tol or time.perf_counter() - start > time_limit:
            break
        history.append(res)
        if (len(history) > stall_window and len(x) > 1
                and res > stall_ratio * history[-stall_window - 1]):
            saved.append((x.copy(), w.copy(), float(v)))
            x, w = _merge_pair(x, w, int(np.argmin(np.diff(x))))
            forced += 1
            history = []
            continue
        J = _jacobian(x, w, fixed, p, tau, rho)
        JtJ, g = J.T @ J, J.T @ f
        for _ in range(30):  # Levenberg-Marquardt step with adaptive damping
            d = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-30), -g)
            zn = z + d
            xn = x.copy()
            xn[~fixed] = zn[:nf]
            if xn.min() >= 0 and xn.max() <= 1 and np.linalg.norm(F(zn)) < np.linalg.norm(f):
                lam = max(lam / 3, 1e-12)
                break
            lam *= 4
        else:
            break  # no descent step: stop
        x = xn
        w, v = zn[nf:-1], zn[-1]
    # choose among the final state and the states saved before forced merges
    cands = saved + [(x, w, float(v))]
    if len(cands) > max_checks:  # evaluate at most max_checks of them (first and last kept)
        idx = np.unique(np.linspace(0, len(cands) - 1, max_checks).astype(int))
        cands = [cands[i] for i in idx]
    gp = dict(corr=rho, noise=tau, R=1, Z=0, P=p)
    gaps = []
    for xc, wc, _ in cands:
        wc = np.clip(wc, 0, None) / np.clip(wc, 0, None).sum()
        vc = wc @ expected_rewards(xc, xc, wc, gp)
        # own scan (seed 2): the Double Oracle oracle uses seed 0, so the start
        # would look converged on its own points; the final check uses seed 1
        gaps.append(continuous_best_response(gp, xc, wc, log_points=14, refine=8,
                                             seed=2)[0] - vc)
    k = int(np.argmin(gaps))
    x, w, v = cands[k]
    w = np.clip(w, 0, None) / np.clip(w, 0, None).sum()
    res = float(np.abs(_system(x, w, v, x < 1e-12, p, tau, rho)).max())
    return x, w, float(v), {"iterations": it, "merges": merges, "forced_merges": forced,
                            "drops": drops, "residual": res, "atoms": len(x),
                            "chosen": ("final" if k == len(cands) - 1 else
                                       "start" if k == 0 else "before a forced merge"),
                            "time": time.perf_counter() - start}


def fp_start(p=1.0, tau=1e-3, rho=0.5, n=2048, time_limit=60, thresh=1e-4):
    """Fictitious play on the n-grid (baselines.fictitious_play), last snapshot;
    returns its points with weight above thresh * max, for newton_atoms."""
    from baselines import fictitious_play

    gp = dict(corr=rho, noise=tau, R=1, Z=0, P=p)
    x = np.linspace(0, 1, n)
    A = reward_matrix(x, x, **gp)
    snaps = fictitious_play(A, x, time_limit=time_limit)
    q = (snaps[-1][2][0] + snaps[-1][2][1]) / 2
    keep = q > thresh * q.max()
    return x[keep], q[keep] / q[keep].sum(), {"rounds": int(snaps[-1][0]),
                                              "points": int(keep.sum())}
