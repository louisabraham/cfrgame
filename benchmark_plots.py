"""
Plots and table for benchmark.py: accuracy (QuasiNashConv) vs running time.
Each method is one curve: the best trade-off (lower envelope) over all its
(grid size n, iterations) points.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import ANYTIME, GAME, METHODS, OUT, TIME_LIMIT, TIMEOUT, matrix_time

HERE = OUT.parent.parent
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
FAMILIES = [
    ("ours", "Our solver: same grid vs shifted grids"),
    ("exact", "Exact grid solvers (LCP, QP)"),
    ("nashopt", "nashopt"),
    ("local", "Local / Newton"),
    ("support", "Support generation"),
    ("dynamics", "Learning dynamics"),
]
FLOOR = 1e-16  # QNC values are clipped here for the log scale


def reason(fail_text):
    if fail_text.startswith("timeout"):
        return "timeout"
    if "size-limited license" in fail_text:
        return "Gurobi license size limit"
    if "equilibria[0]" in fail_text and "IndexError" in fail_text:
        return "no equilibrium returned"
    if "NaN" in fail_text:
        return "NaN output"
    if "no solution" in fail_text:
        return "no solution in time limit"
    return "error"


def load():
    table = json.loads((OUT / "results.json").read_text())
    runs = {}
    for key, d in table.items():
        if key == "analytical" or "qnc" not in d:
            continue
        if METHODS[d["method"]][2] == "grid":
            d["t_matrix"] = matrix_time(d["n"])
        d["time"] = d["t_solve"] + d["t_matrix"]
        runs.setdefault(d["method"], []).append(d)
    for rs in runs.values():
        rs.sort(key=lambda d: d["n"])
    fails = {}
    for d in sorted((d for d in table.values() if "failed" in d), key=lambda d: -d["n"]):
        fails[d["method"]] = d  # keep the smallest failing n
    for m, d in fails.items():  # drop results above the first failure
        if m in runs:
            runs[m] = [r for r in runs[m] if r["n"] < d["n"]]
    return runs, fails, table.get("analytical")


def points(rs):
    """All (time, qnc, n, iterations) points of a method; iterations is None
    for methods without an iteration count."""
    out = []
    for d in rs:
        for snap in d.get("snapshots", [d]):
            out.append((d["t_matrix"] + snap["t_solve"], snap["qnc"], d["n"],
                        snap.get("iters") if "snapshots" in d else None))
    return out


def front(pts):
    """Lower envelope: the points that no faster point beats in accuracy."""
    out, best = [], np.inf
    for p in sorted(pts, key=lambda p: (p[0], p[1])):
        if p[1] < best:
            out.append(p)
            best = p[1]
    return out


def label(p):
    return f"n={p[2]}" + ("" if p[3] is None else f", {p[3]} it")


def _curve(ax, runs, method, color, marker, ls="-", lw=1.5, z=3, annotate=False):
    rs = runs.get(method)
    if not rs:
        return
    f = front(points(rs))
    t = [p[0] for p in f]
    q = [max(p[1], FLOOR) for p in f]
    ax.plot(t, q, ls=ls, lw=lw, color=color, marker=marker, ms=4.5,
            mec=SURFACE, mew=0.8, label=METHODS[method][0], zorder=z,
            drawstyle="steps-post")
    if annotate:
        for p in f[:: max(1, len(f) // 6)]:
            ax.annotate(label(p), (p[0], max(p[1], FLOOR)), xytext=(4, 3),
                        textcoords="offset points", fontsize=6, color=MUTED, zorder=z)


def _style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def plot():
    runs, fails, analytical = load()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (family, title) in zip(axes.flat, FAMILIES):
        _style(ax)
        _curve(ax, runs, "rm", INK, "o", z=5, annotate=family == "ours")
        _curve(ax, runs, "cfr", INK2, "o", ls="--", z=4)
        members = [m for m, v in METHODS.items() if v[1] == family and m not in ("rm", "cfr")]
        for k, m in enumerate(members):
            _curve(ax, runs, m, SERIES[k], MARKERS[k])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="lower left", labelcolor=INK2)
    notes = ("Each curve: best QuasiNashConv reachable in a given time, over all grid sizes n "
             "and iteration counts (lower envelope). "
             f"1 thread per solve, limit {TIME_LIMIT} s, hard timeout {TIMEOUT} s. "
             "Grey labels: n and iterations of the RM points. Continuous Double Oracle: n = points "
             f"scanned per best response. Analytical equilibrium ({analytical['kind']}): "
             f"QNC {analytical['qnc']:.1e}.")
    fig.text(0.01, 0.005, notes, fontsize=8, color=INK2, ha="left", va="bottom", wrap=True)
    qs = [max(d["qnc"], FLOOR) for rs in runs.values() for d in rs]
    axes[0, 0].set_ylim(min(qs) / 3, max(qs) * 3)
    for ax in axes[:, 0]:
        ax.set_ylabel("QuasiNashConv (continuous best response)", fontsize=9, color=INK2)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s): payoff matrix + solve", fontsize=9, color=INK2)
    fig.suptitle(
        rf"Accuracy vs running time, 2-player CfR game ($P={GAME['P']:g}$, "
        rf"$\tau={GAME['noise']:g}$, $\rho={GAME['corr']:g}$), grids $n = 16 \ldots 4096$", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    for ext in ("svg", "png"):
        fig.savefig(HERE / "plots" / f"benchmark_accuracy_time_tau={GAME['noise']:g}.{ext}", dpi=150,
                    facecolor=SURFACE)
    plt.close(fig)
    write_table(runs, fails, analytical)


BUDGETS = [0.01, 0.1, 1, 10, 100, np.inf]  # inf: the whole run (120 s limit)


def budget_label(b):
    return "whole run" if b == np.inf else f"<= {b:g} s"


def write_table(runs, fails, analytical):
    """Best QuasiNashConv within each time budget, with the (n, iterations)
    that reach it; then the full lower envelope of each method."""
    head = "| method | " + " | ".join(budget_label(b) for b in BUDGETS) + " |"
    rows = [head, "|" + "---|" * (len(BUDGETS) + 1)]
    fronts = {m: front(points(runs[m])) for m in METHODS if runs.get(m)}
    for m in METHODS:
        cells = []
        for b in BUDGETS:
            ok = [p for p in fronts.get(m, []) if p[0] <= b]
            cells.append(f"{ok[-1][1]:.1e} ({label(ok[-1])})" if ok else "")
        rows.append(f"| {METHODS[m][0]} | " + " | ".join(cells) + " |")
    stopped = [f"- {METHODS[m][0]}: n={d['n']} ({reason(d['failed'])})"
               for m, d in sorted(fails.items(), key=lambda kv: kv[1]["n"])]
    text = [f"# Benchmark results, P={GAME['P']:g}, tau={GAME['noise']:g}, rho={GAME['corr']:g}", "",
            "Cell: best QuasiNashConv reachable within the time budget (payoff matrix + solve),",
            "over all grid sizes n and iteration counts, with the n and iterations that reach it.",
            "Iterations: CFR, multiplicative weights, replicator, extragradient: iterations;",
            "RM and fictitious play: rounds of n updates (one round costs O(n^2) like a CFR",
            "iteration).", ""] + rows
    text += ["", "First failing grid size per method:", ""] + stopped
    if analytical:
        text += ["", f"Analytical equilibrium ({analytical['kind']}): "
                 f"QuasiNashConv {analytical['qnc']:.1e}."]
    text += ["", "## Lower envelope of each method", ""]
    for m, f in fronts.items():
        pts = ", ".join(f"{p[1]:.1e} at {p[0]:.3g} s ({label(p)})" for p in f)
        text.append(f"- {METHODS[m][0]}: {pts}")
    (OUT / "RESULTS.md").write_text("\n".join(text) + "\n")


def summary(taus=("0.03", "0.0001", "0")):
    """bench/SUMMARY.md: best QuasiNashConv of every method for each tau,
    within 10 s and within the whole run, from bench/tau=<tau>/results.json."""
    root = HERE / "bench"
    best = {}
    for tau in taus:
        d = root / f"tau={tau}"
        table = json.loads((d / "results.json").read_text())
        runs = {}
        for key, r in table.items():
            if key == "analytical" or "qnc" not in r:
                continue
            if METHODS[r["method"]][2] == "grid":
                r["t_matrix"] = json.loads((d / "matrices" / f"A_{r['n']}.json").read_text())["t_matrix"]
            runs.setdefault(r["method"], []).append(r)
        fails = {}
        for r in sorted((r for r in table.values() if "failed" in r), key=lambda r: -r["n"]):
            fails[r["method"]] = r
        for m, f in fails.items():
            if m in runs:
                runs[m] = [r for r in runs[m] if r["n"] < f["n"]]
        for m, rs in runs.items():
            fr = front(points(rs))
            for b in (10, np.inf):
                ok = [p for p in fr if p[0] <= b]
                best[m, tau, b] = ok[-1] if ok else None
    head = "| method | " + " | ".join(f"tau={t} <= 10 s | tau={t} whole run" for t in taus) + " |"
    rows = [head, "|" + "---|" * (2 * len(taus) + 1)]

    def cell(p):
        return "" if p is None else f"{p[1]:.1e} ({p[0]:.3g} s, {label(p)})"

    # rank by the best whole-run QNC at the first tau
    order = sorted(METHODS, key=lambda m: (best.get((m, taus[0], np.inf)) or (0, np.inf))[1])
    for m in order:
        cells = [cell(best.get((m, t, b))) for t in taus for b in (10, np.inf)]
        rows.append(f"| {METHODS[m][0]} | " + " | ".join(cells) + " |")
    text = ["# Best QuasiNashConv per method (P=1, rho=0.5)", "",
            "Best over all grid sizes n and iteration counts, within 10 s and within the",
            "whole run (120 s solver limit, hard timeout 240 s). Cell: QNC (time, n, iterations).",
            "Sorted by the whole-run result at the first tau.", ""] + rows
    (root / "SUMMARY.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    import sys

    summary() if sys.argv[1:] == ["summary"] else plot()
