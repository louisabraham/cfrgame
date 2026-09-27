"""Build the LaTeX tables of bench/report/report.tex from the result files,
then compile the PDF.

    python bench/scripts/make_report.py   ->  bench/report/report.pdf
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import METHODS  # noqa: E402
from benchmark_plots import front, points  # noqa: E402

BENCH = ROOT / "bench"
REPORT = BENCH / "report"
TABLES = REPORT / "tables"
TAUS = ["0.03", "0.0001", "0"]
TAU_TEX = {"0.03": "0.03", "0.0001": "10^{-4}", "0": "0"}


def sci(v, digits=1):
    """2.3e-05 -> $2.3\\cdot10^{-5}$"""
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "$0$"
    m, e = f"{v:.{digits}e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


def secs(t):
    return f"{t:.2g}" if t < 10 else f"{t:.0f}"


def load(tau):
    """Runs (with snapshots) and first failures of one tau, as in benchmark_plots."""
    d = BENCH / f"tau={tau}"
    table = json.loads((d / "results.json").read_text())
    runs, fails = {}, {}
    for key, r in table.items():
        if key == "analytical" or "qnc" not in r:
            continue
        if METHODS[r["method"]][2] == "grid":
            r["t_matrix"] = json.loads((d / "matrices" / f"A_{r['n']}.json").read_text())["t_matrix"]
        runs.setdefault(r["method"], []).append(r)
    for r in sorted((r for r in table.values() if "failed" in r), key=lambda r: -r["n"]):
        fails[r["method"]] = r
    for m, f in fails.items():
        if m in runs:
            runs[m] = [r for r in runs[m] if r["n"] < f["n"]]
    return runs, fails, table


def best(runs, m, budget):
    if not runs.get(m):
        return None
    ok = [p for p in front(points(runs[m])) if p[0] <= budget]
    return ok[-1] if ok else None


def where(p):
    return f"$n={p[2]}$" + ("" if p[3] is None else f", {p[3]} it.")


def summary_tables():
    data = {tau: load(tau)[0] for tau in TAUS}
    order = sorted(METHODS, key=lambda m: (best(data["0.03"], m, np.inf) or (0, np.inf))[1])
    head = " & ".join(f"$\\tau={TAU_TEX[t]}$" for t in TAUS)
    rows_all, rows_10 = [], []
    for m in order:
        name = METHODS[m][0].replace("&", "\\&")
        cells_all, cells_10 = [], []
        for t in TAUS:
            p = best(data[t], m, np.inf)
            cells_all.append("--" if p is None else f"{sci(p[1])} ({secs(p[0])} s)")
            q = best(data[t], m, 10)
            cells_10.append("--" if q is None else sci(q[1]))
        rows_all.append(f"{name} & " + " & ".join(cells_all) + r" \\")
        rows_10.append(f"{name} & " + " & ".join(cells_10) + r" \\")
    for fname, rows, cols in [("summary_all.tex", rows_all, "lccc"),
                              ("summary_10s.tex", rows_10, "lccc")]:
        (TABLES / fname).write_text(
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\nMethod & {head} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def support_exact():
    """Double Oracle supports against the exact 6 atoms (tau = 0.03)."""
    from analytical import solve

    eq = solve(1.0, 0.03, 0.5)
    A, W = eq.atoms, eq.weights
    table = json.loads((BENCH / "tau=0.03" / "results.json").read_text())
    rows = []
    for m, label in [("do_cont", "continuous"), ("do_grid", "grid")]:
        for n in [64, 256, 1024, 4096]:
            k = f"{m}_{n}"
            d = np.load(BENCH / "tau=0.03" / "runs" / f"{k}.npz")
            a, p = d["a1"], d["p1"]
            pos = p > 1e-9
            ap, pp = a[pos], p[pos]
            j = np.abs(ap[:, None] - A[None, :]).argmin(1)
            mass = np.bincount(j, weights=pp, minlength=len(A))
            cnt = np.bincount(j, minlength=len(A))
            cen = np.array([(ap[j == i] * pp[j == i]).sum() / max(mass[i], 1e-300)
                            for i in range(len(A))])
            rows.append(f"{label} & {n} & {sci(table[k]['qnc'])} & "
                        f"{table[k]['info'].get('support', len(a))} & {int(pos.sum())} & "
                        f"{','.join(map(str, cnt))} & {sci(np.abs(cen - A)[cnt > 0].max())} & "
                        f"{sci(np.abs(mass - W).max())} \\\\")
    (TABLES / "support_exact.tex").write_text(
        "\\begin{tabular}{lrcrrlcc}\n\\toprule\n"
        "DO & $n$ & QNC & points added & weight $>0$ & points per atom & position error & weight error \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return A, W


def support_small_tau():
    """Support of CFR, fictitious play and Double Oracle at tau = 1e-4 (n = 4096
    for the iterative methods, best result of each Double Oracle)."""
    runs, _, table = load("0.0001")
    rows = []
    for m in ["fp", "cfr", "do_grid", "do_cont"]:
        rs = runs.get(m, [])
        if not rs:
            continue
        r = min(rs, key=lambda r: min(s["qnc"] for s in r.get("snapshots", [r])))
        d = np.load(BENCH / "tau=0.0001" / "runs" / f"{m}_{r['n']}.npz")
        if "snapshots" in r:
            j = int(np.argmin([s["qnc"] for s in r["snapshots"]]))
            p, qnc = d["snap_p1"][j], r["snapshots"][j]["qnc"]
        else:
            p, qnc = d["p1"], r["qnc"]
        a = d["a1"]
        keep = p > 1e-9
        order = np.argsort(-p[keep])
        k999 = int(np.searchsorted(np.cumsum(p[keep][order]), 0.999) + 1)
        s = np.sort(a[keep][order[:k999]])
        rows.append(f"{METHODS[m][0]} & {r['n']} & {sci(qnc)} & {int(keep.sum())} & {k999} & "
                    f"$[{s.min():.3f}, {s.max():.3f}]$ \\\\")
    (TABLES / "support_small_tau.tex").write_text(
        "\\begin{tabular}{lrcrrc}\n\\toprule\n"
        "Method & $n$ & QNC & weight $>10^{-9}$ & 99.9\\% of the mass & range \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def refine_table():
    rows = []
    for tau in ["0.03", "0.01", "0.001", "0.0001"]:
        for start, label in [("do", "Double Oracle"), ("fp", "fictitious play")]:
            f = BENCH / "refine" / f"{start}_tau{tau}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            rows.append(f"${float(tau):g}$ & {label} & {sci(d['start_qnc'])} & {secs(d['start_time'])} & "
                        f"{sci(d['qnc'])} & {secs(d['time'])} & {d['atoms']} & "
                        f"{d['merges']}+{d['forced_merges']} \\\\")
    (TABLES / "refine.tex").write_text(
        "\\begin{tabular}{llcrcrrr}\n\\toprule\n"
        "$\\tau$ & start & QNC start & time (s) & QNC after & Newton (s) & atoms & merges \\\\\n"
        "\\midrule\n" + ("\n".join(rows) if rows else "\\multicolumn{8}{c}{not run} \\\\")
        + "\n\\bottomrule\n\\end{tabular}\n")


def multiplayer_table():
    rows = []
    for n in [2, 3, 5, 10]:
        f = BENCH / "multiplayer" / f"n{n}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rows.append(f"{n} & {sci(d['nashconv'])} & {sci(d['nashconv_check'])} & {secs(d['time'])} & "
                    f"{d['iterations']} & {d['X']} & {d['support']} & {len(d['atoms'])} & "
                    f"{d['weights'][0]:.3f} \\\\")
    (TABLES / "multiplayer.tex").write_text(
        "\\begin{tabular}{rccrrrrrc}\n\\toprule\n"
        "$n$ & NashConv & check & time (s) & iterations & $|X|$ & weight $>0$ & atoms & mass at 0 \\\\\n"
        "\\midrule\n" + ("\n".join(rows) if rows else "\\multicolumn{9}{c}{not run} \\\\")
        + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    TABLES.mkdir(parents=True, exist_ok=True)
    summary_tables()
    support_exact()
    support_small_tau()
    refine_table()
    multiplayer_table()
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "report.tex"],
                   cwd=REPORT, check=True)
    print("wrote", REPORT / "report.pdf")
