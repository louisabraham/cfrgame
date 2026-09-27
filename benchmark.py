"""
Compare our regret-matching solver with other equilibrium solvers on the
2-player CfR game with friction and correlation, for several grid sizes.

    python benchmark.py run        # solve every (method, grid) pair
    python benchmark.py evaluate   # QuasiNashConv of every stored solution
    python benchmark.py plot       # plots/benchmark_*.svg + table

The game is P=1, rho=0.5 and tau=$BENCH_TAU (default 0.03); results go to
bench/tau=<tau>/.

Protocol
- Every grid method uses the same grid linspace(0, 1, n) for both players.
- Every solve runs in its own process with 1 thread (BLAS, numba, Gurobi,
  HiGHS, XLA) and a hard timeout, at most 10 in parallel (fewer than the
  performance cores), on AC power only. For a method, the grid sizes run in
  increasing order and stop after the first failure or timeout. Each timed
  solve follows an untimed warm-up solve on an 8-point grid (JIT, imports).
- Time = payoff-matrix construction (measured once per n, single thread)
  + solver time. Iterative methods run once per n with snapshots after
  chunks of doubling size; each snapshot is one (n, iterations) point, and
  the plots keep the best trade-off (lower envelope) over all points. Double Oracle builds no full matrix; for the continuous
  variant, n is the number of points scanned by each best response.
- Quality = QuasiNashConv with continuous best responses over r in [0, 1]
  (Sobol scan + Brent refinement), not the regret on the grid.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
TAU = float(os.environ.get("BENCH_TAU", "0.03"))  # select the game: BENCH_TAU=0 ...
GAME = dict(corr=0.5, noise=TAU, R=1, Z=0, P=1.0)
OUT = HERE / "bench" / f"tau={TAU:g}"
CACHE = OUT / "matrices"
GRIDS = [2**k for k in range(4, 13)]  # 16 ... 4096
TIMEOUT = 240  # hard kill (s)
TIME_LIMIT = 120  # solver time limit for exact / global methods (s)
BUDGET = TIME_LIMIT  # time limit of the iterative methods (s)
# iterative methods: one run per n, snapshots after chunks of 1, 1, 2, 4, 8, ...
# iterations; every snapshot is one (n, iterations) point of the comparison
ANYTIME = {"rm", "cfr", "rm_shift", "cfr_shift", "fp", "mwu", "replicator",
           "nashopt_extragrad"}

# name: (label, family, kind); kind "grid" uses the n-grid payoff matrix
METHODS = {
    "rm": ("RM (ours)", "ours", "grid"),
    "cfr": ("CFR (ours)", "ours", "grid"),
    # the paper's setting: interleaved grids for the two players (no exact ties)
    "rm_shift": ("RM (ours), shifted grids", "ours", "shifted"),
    "cfr_shift": ("CFR (ours), shifted grids", "ours", "shifted"),
    "gambit_lh": ("Lemke-Howson (Gambit)", "exact", "grid"),
    "lh_tableau": ("Lemke-Howson (numpy tableau)", "exact", "grid"),
    "gambit_logit": ("Logit QRE path (Gambit)", "exact", "grid"),
    "ms_gurobi": ("Mangasarian-Stone QP (Gurobi)", "exact", "grid"),
    "gurobi_qp": ("Nash-gap QP, 2n bilinear (Gurobi)", "exact", "grid"),
    "scip_qp": ("Nash-gap QP, 2n bilinear (SCIP)", "exact", "grid"),
    "fb_ncp": ("FB-NCP Newton, symmetric", "local", "grid"),
    "do_grid": ("Double Oracle, grid BR", "support", "grid"),
    # no payoff matrix; n = number of Sobol points scanned by the best response
    "do_cont": ("Double Oracle, continuous BR", "support", "continuous"),
    # continuous Double Oracle (half the time), then hybrid.newton_atoms: moves
    # the atoms, merges pairs (tau > 0 only)
    "do_newton": ("Double Oracle + Newton on atoms", "support", "continuous"),
    "ipopt_ms": ("Ipopt multi-start", "local", "grid"),
    "fp": ("Fictitious play", "dynamics", "grid"),
    "mwu": ("Multiplicative weights", "dynamics", "grid"),
    "replicator": ("Replicator dynamics", "dynamics", "grid"),
    "nashopt_milp_highs": ("nashopt MILP (HiGHS)", "nashopt", "grid"),
    "nashopt_milp_gurobi": ("nashopt MILP (Gurobi)", "nashopt", "grid"),
    "nashopt_lemke": ("nashopt Lemke", "nashopt", "grid"),
    "nashopt_gnep": ("nashopt GNEP (FB least squares)", "nashopt", "grid"),
    "nashopt_extragrad": ("nashopt extragradient", "nashopt", "grid"),
    "nashopt_drdaqp": ("nashopt DR-DAQP", "nashopt", "grid"),
}

SINGLE_THREAD = {
    k: "1"
    for k in [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
    ]
}
SINGLE_THREAD["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
SINGLE_THREAD["JAX_PLATFORMS"] = "cpu"


def result_path(method, n):
    return OUT / "runs" / f"{method}_{n}.npz"


def grid(n):
    return np.linspace(0, 1, n)


def matrix_time(n):
    return json.loads((CACHE / f"A_{n}.json").read_text())["t_matrix"]


def matrix(n):
    """Cached payoff matrix of the n-grid and its single-thread build time."""
    from game import reward_matrix

    path = CACHE / f"A_{n}.npy"
    meta = CACHE / f"A_{n}.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        reward_matrix(grid(4), grid(4), **GAME)  # JIT outside the timing
        t = time.perf_counter()
        A = reward_matrix(grid(n), grid(n), **GAME)
        meta.write_text(json.dumps({"t_matrix": time.perf_counter() - t}))
        np.save(path, A)
    return np.load(path), json.loads(meta.read_text())["t_matrix"]


# ---------------------------------------------------------------------------
# one solve (runs in a subprocess)
# ---------------------------------------------------------------------------


def solve_one(method, n):
    import baselines as bl
    from game import gen_actions, reward_matrix

    shifted = {}

    def do_newton(n, limit):
        import hybrid as hy

        if GAME["noise"] <= 0:
            raise RuntimeError("Newton on atoms needs tau > 0 (smooth payoff)")
        a, q1, _, q2, info = bl.double_oracle(
            GAME, grid=None, time_limit=limit / 2, log_points=int(np.log2(n)))
        x, w, _, ninfo = hy.newton_atoms(a, (q1 + q2) / 2, GAME["P"], GAME["noise"],
                                         GAME["corr"], time_limit=limit / 2)
        return x, w, x, w, {"do": info, "newton": ninfo}

    def ours(A, x, cfr, budget, shift=False):
        if shift:  # interleaved grids: two matrices, built and timed here
            a1, a2 = gen_actions(len(x), True)
            t = time.perf_counter()
            A1, A2 = reward_matrix(a1, a2, **GAME), reward_matrix(a2, a1, **GAME)
            shifted["t_matrix"], shifted["a"] = time.perf_counter() - t, (a1, a2)
        else:
            a1 = a2 = x
            A1 = A2 = A
        return bl.regret_matching_anytime(A1, A2, a1, a2, cfr=cfr, time_limit=budget)

    def call(A, x, n, limit, budget):
        return {
            "rm": lambda: ours(A, x, False, budget),
            "cfr": lambda: ours(A, x, True, budget),
            "rm_shift": lambda: ours(A, x, False, budget, shift=True),
            "cfr_shift": lambda: ours(A, x, True, budget, shift=True),
            "gambit_lh": lambda: bl.gambit_lh(A, x),
            "lh_tableau": lambda: bl.lh_tableau(A, x),
            "gambit_logit": lambda: bl.gambit_logit(A, x),
            "ms_gurobi": lambda: bl.mangasarian_stone(A, x, time_limit=limit),
            "gurobi_qp": lambda: bl.gurobi_qp(A, x, time_limit=limit),
            "scip_qp": lambda: bl.scip_qp(A, x, time_limit=limit),
            "fb_ncp": lambda: bl.fb_ncp(A, x, time_limit=limit),
            "do_grid": lambda: bl.double_oracle(GAME, grid=x, time_limit=limit),
            "do_cont": lambda: bl.double_oracle(
                GAME, grid=None, time_limit=limit, log_points=int(np.log2(n))),
            "do_newton": lambda: do_newton(n, limit),
            "ipopt_ms": lambda: bl.ipopt_ms(A, x, time_limit=limit),
            "fp": lambda: bl.fictitious_play(A, x, time_limit=budget),
            "mwu": lambda: bl.mwu(A, x, time_limit=budget),
            "replicator": lambda: bl.replicator(A, x, time_limit=budget),
            "nashopt_milp_highs": lambda: bl.nashopt_milp(A, x, "highs", limit),
            "nashopt_milp_gurobi": lambda: bl.nashopt_milp(A, x, "gurobi", limit),
            "nashopt_lemke": lambda: bl.nashopt_lemke(A, x),
            "nashopt_gnep": lambda: bl.nashopt_gnep(A, x),
            "nashopt_extragrad": lambda: bl.nashopt_extragrad(A, x, time_limit=budget),
            "nashopt_drdaqp": lambda: bl.nashopt_drdaqp(A, x),
        }[method]()

    # untimed warm-up on a tiny grid: JIT compilation and lazy imports
    w = grid(8)
    call(reward_matrix(w, w, **GAME), w, 8, 5, 0.5)

    x = grid(n)
    kind = METHODS[method][2]
    A, t_matrix = matrix(n) if kind == "grid" else (None, 0.0)
    if method in ANYTIME:
        snaps = call(A, x, n, TIME_LIMIT, BUDGET)
        a1, a2 = shifted.get("a", (x, x))
        t_matrix = shifted.get("t_matrix", t_matrix)
        np.savez(
            result_path(method, n), a1=a1, a2=a2,
            p1=snaps[-1][2][0], p2=snaps[-1][2][1],
            snap_p1=np.array([s[2][0] for s in snaps]),
            snap_p2=np.array([s[2][1] for s in snaps]),
            snap_iters=np.array([s[0] for s in snaps]),
            snap_times=np.array([s[1] for s in snaps]),
            meta=json.dumps({"method": method, "n": n, "t_solve": snaps[-1][1],
                             "t_matrix": t_matrix, "iters": int(snaps[-1][0]),
                             "info": {}}),
        )
        return
    t = time.perf_counter()
    a1, p1, a2, p2, info = call(A, x, n, TIME_LIMIT, BUDGET)
    t_solve = time.perf_counter() - t
    np.savez(
        result_path(method, n), a1=a1, p1=p1, a2=a2, p2=p2,
        meta=json.dumps({"method": method, "n": n, "t_solve": t_solve,
                         "t_matrix": t_matrix, "info": info}, default=str),
    )


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_chain(method, grids, log):
    for n in grids:
        path = result_path(method, n)
        fail = path.with_suffix(".fail")
        if path.exists():
            continue
        if fail.exists():
            return
        t = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "solve", method, str(n)],
                env={**os.environ, **SINGLE_THREAD}, cwd=HERE, timeout=TIMEOUT,
                capture_output=True, text=True,
            )
            status = "ok" if proc.returncode == 0 else f"error: {proc.stderr[-600:]}"
        except subprocess.TimeoutExpired:
            status = f"timeout after {TIMEOUT} s"
        if not on_ac_power():
            status = "error: stopped, on battery power"
        log(f"{method:20s} n={n:5d} {time.perf_counter() - t:7.1f}s {status.splitlines()[0] if status else ''}")
        if not on_ac_power():
            path.unlink(missing_ok=True)  # the time of this solve is not valid
            sys.exit("on battery power: benchmark stopped")
        if status != "ok":
            fail.write_text(status)
            return


def on_ac_power():
    if sys.platform != "darwin":
        return True
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "AC Power" in out


def run(methods, grids, jobs):
    # timings are only comparable on AC power (macOS slows the CPU on battery)
    if not on_ac_power():
        sys.exit("on battery power: plug in before running the benchmark")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    # build the payoff matrices first, one at a time, so that their build
    # times are measured without load
    for n in grids:
        if not (CACHE / f"A_{n}.npy").exists():
            subprocess.run([sys.executable, __file__, "matrix", str(n)],
                           env={**os.environ, **SINGLE_THREAD}, cwd=HERE, check=True)
    logfile = open(OUT / "run.log", "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda m: run_chain(m, grids, log), methods))


# ---------------------------------------------------------------------------
# evaluation (all cores)
# ---------------------------------------------------------------------------


def quasinashconv(a1, p1, a2, p2):
    """Continuous-action NashConv: sum over players of max_r E[u(r, opp)] - value."""
    from baselines import continuous_best_response, expected_rewards

    k1, k2 = p1 > 0, p2 > 0
    a1, p1, a2, p2 = a1[k1], p1[k1], a2[k2], p2[k2]
    v1 = p1 @ expected_rewards(a1, a2, p2, GAME)
    v2 = p2 @ expected_rewards(a2, a1, p1, GAME)
    br1, _ = continuous_best_response(GAME, a2, p2, log_points=14, refine=8, seed=1)
    symmetric = len(a1) == len(a2) and np.array_equal(a1, a2) and np.array_equal(p1, p2)
    br2 = br1 if symmetric else continuous_best_response(
        GAME, a1, p1, log_points=14, refine=8, seed=1)[0]
    return float(br1 - v1 + br2 - v2)


def evaluate():
    from baselines import grid_gap

    table_path = OUT / "results.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {}
    for path in sorted((OUT / "runs").glob("*.npz")):
        if path.stem in table:
            continue
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        if not (np.isfinite(d["p1"]).all() and np.isfinite(d["p2"]).all()):
            table[path.stem] = {"method": meta["method"], "n": meta["n"],
                                "failed": "error: invalid output (NaN strategy)"}
            continue
        t = time.perf_counter()
        if "snap_iters" in d:
            meta["snapshots"] = [
                {"iters": int(k), "t_solve": float(ts),
                 "qnc": quasinashconv(d["a1"], q1, d["a2"], q2)}
                for k, ts, q1, q2 in zip(d["snap_iters"], d["snap_times"],
                                         d["snap_p1"], d["snap_p2"])]
            meta["qnc"] = meta["snapshots"][-1]["qnc"]
        else:
            meta["qnc"] = quasinashconv(d["a1"], d["p1"], d["a2"], d["p2"])
        if METHODS[meta["method"]][2] == "grid":
            meta["grid_nashconv"] = float(grid_gap(matrix(meta["n"])[0], d["p1"], d["p2"]))
        meta["support"] = int(max((d["p1"] > 1e-9).sum(), (d["p2"] > 1e-9).sum()))
        table[path.stem] = meta
        print(f"{path.stem:28s} QNC={meta['qnc']:.3e}  ({time.perf_counter() - t:.1f}s)", flush=True)
        table_path.write_text(json.dumps(table, indent=1))
    for path in sorted((OUT / "runs").glob("*.fail")):
        method, n = path.stem.rsplit("_", 1)
        table[path.stem] = {"method": method, "n": int(n), "failed": path.read_text()[-400:]}
    # reference: the analytical equilibrium of analytical.py
    from analytical import solve

    try:
        eq = solve(GAME["P"], GAME["noise"], GAME["corr"])
    except RuntimeError as e:  # e.g. small tau: too many atoms for the solver
        table["analytical"] = {"method": "analytical", "kind": f"not available: {e}",
                               "qnc": float("nan")}
        table_path.write_text(json.dumps(table, indent=1))
        return
    if eq.kind == "atomic":
        atoms, weights, what = eq.atoms, eq.weights, "atoms"
    else:
        # atomless density: equal-mass quantile atoms (bin midpoints)
        k = 2**13
        F = np.concatenate([[0.0], np.cumsum(
            0.5 * np.diff(eq.grid) * (eq.density[1:] + eq.density[:-1]))])
        atoms = np.interp((np.arange(k) + 0.5) / k, F / F[-1], eq.grid)
        weights, what = np.full(k, 1.0 / k), f"density, {k} quantile atoms"
    table["analytical"] = {"method": "analytical", "kind": what,
                           "atoms": atoms.tolist() if len(atoms) < 50 else None,
                           "qnc": quasinashconv(atoms, weights, atoms, weights)}
    table_path.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("method")
    s.add_argument("n", type=int)
    s = sub.add_parser("matrix")
    s.add_argument("n", type=int)
    s = sub.add_parser("run")
    s.add_argument("--methods", nargs="*", default=list(METHODS))
    s.add_argument("--grids", nargs="*", type=int, default=GRIDS)
    # at most the number of performance cores minus 2 (M4 Max: 12 P + 4 E cores)
    s.add_argument("--jobs", type=int, default=10)
    sub.add_parser("evaluate")
    sub.add_parser("plot")
    args = ap.parse_args()
    if args.cmd == "solve":
        try:
            solve_one(args.method, args.n)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    elif args.cmd == "matrix":
        matrix(args.n)
    elif args.cmd == "run":
        run(args.methods, args.grids, args.jobs)
    elif args.cmd == "evaluate":
        evaluate()
    elif args.cmd == "plot":
        from benchmark_plots import plot

        plot()
