"""
Compare the analytical equilibria (analytical.py, following analytical.md)
with the regret-matching solver, one example in every regime of the phase
diagram, for R = 1, Z = 0 and P in {0, 1}.

Produces plots/analytical_vs_rm.svg/.png and prints a summary table.
For tau = 0 the panels show the equilibrium density f(a); for tau > 0 the
equilibrium is atomic, so the panels show probability mass per action.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytical import solve, tau0, tau_c
from regret_matching import regret_matching

# FP flags raised inside numba/cython code get attributed by numpy to the
# next checked operation; matmuls of finite arrays cannot raise these.
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)

ACTIONS = 2**12
ITERS = 10000  # regret matching multiplies this by ACTIONS -> 4.1e7 updates

# palette: RM players are series colors, the analytical curve is primary ink
C_RM1, C_RM2, C_ANA = "#2a78d6", "#008300", "#0b0b0b"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def examples():
    out = []
    for P in (0.0, 1.0):
        t0, tc = tau0(P), tau_c(P)
        # one example per regime; tau strictly inside each phase
        out += [
            dict(P=P, tau=0.0, rho=0.0, label="$\\tau=0$, $\\rho=0$\ncontinuous (closed form)"),
            dict(P=P, tau=0.0, rho=0.5, label="$\\tau=0$, $\\rho=0.5$\ncontinuous (shooting)"),
            dict(P=P, tau=round(1.2 * t0, 2), rho=0.0, label="zero-risk pure"),
            dict(P=P, tau=round(0.5 * (tc + t0), 2), rho=0.0, label="positive pure"),
            dict(P=P, tau=0.1, rho=0.0, label="atomic mixed"),
            # both friction and correlation nonzero (analytical.md section 6)
            dict(P=P, tau=0.1, rho=-0.5, label="combined"),
            dict(P=P, tau=0.1, rho=0.5, label="combined"),
        ]
    return out


def run_rm(P, tau, rho, seed=0):
    params = dict(corr=rho, noise=tau, R=1, Z=0, P=P)
    _, (a1, p1), (a2, p2), r1, r2, _ = regret_matching(
        params, actions=ACTIONS, iters=ITERS, shift=True, progress=False, seed=seed
    )
    return dict(
        a1=a1, p1=p1, a2=a2, p2=p2,
        mean1=float(a1 @ p1), mean2=float(a2 @ p2),
        u_total=float(p1 @ r1 @ p2 + p2 @ r2 @ p1),
    )


def compute_all():
    results = []
    for ex in examples():
        print(f"P={ex['P']:g} tau={ex['tau']:g} rho={ex['rho']:g} "
              f"({ex['label'].splitlines()[-1]})")
        eq = solve(ex["P"], ex["tau"], ex["rho"])
        rm = run_rm(ex["P"], ex["tau"], ex["rho"])
        results.append((ex, eq, rm))
    return results


def rm_quantile(a, p, q):
    return float(a[np.searchsorted(np.cumsum(p), q)])


def binned_density(a, p, xmax, nbins=400):
    """RM probability mass -> density estimate, binned for display."""
    nbins = min(nbins, max(50, int(round(xmax / (a[1] - a[0]) / 2))))
    edges = np.linspace(0.0, xmax, nbins + 1)
    idx = np.digitize(a, edges) - 1
    valid = (idx >= 0) & (idx < nbins)
    mass = np.bincount(idx[valid], weights=p[valid], minlength=nbins)
    return 0.5 * (edges[:-1] + edges[1:]), mass / (edges[1] - edges[0])


def panel(ax, eq, rm):
    support_max = eq.h if eq.kind == "continuous" else float(eq.atoms.max())
    xm = 1.15 * max(
        support_max,
        rm_quantile(rm["a1"], rm["p1"], 0.999),
        rm_quantile(rm["a2"], rm["p2"], 0.999),
        0.025,
    )
    x0 = -0.04 * xm
    series = [
        (rm["a1"], rm["p1"], C_RM1, "RM player 1"),
        (rm["a2"], rm["p2"], C_RM2, "RM player 2"),
    ]

    if eq.kind == "continuous":
        ymax = eq.density.max()
        for a, p, color, name in series:
            centers, dens = binned_density(a, p, xm)
            ax.plot(centers, dens, color=color, lw=1.0, label=name)
            ymax = max(ymax, float(dens.max()))
        ax.plot(eq.grid, eq.density, color=C_ANA, lw=1.8, ls=(0, (4, 2)),
                label="analytical")
        ax.plot([eq.h, eq.h, xm], [eq.density[-1], 0.0, 0.0],
                color=C_ANA, lw=1.8, ls=(0, (4, 2)))
        text_right = False  # the density peaks at h, on the right
    else:
        # RM smears each atom over neighboring grid points: show the raw
        # masses (vlines) plus, as dots, the RM mass aggregated around each
        # analytical atom, directly comparable to the atom weight
        bounds = np.r_[-np.inf, 0.5 * (eq.atoms[1:] + eq.atoms[:-1]), np.inf]
        ymax = float(eq.weights.max())
        for k, (a, p, color, name) in enumerate(series):
            ax.vlines(a, 0.0, p, color=color, lw=1.0, label=name)
            agg = [float(p[(a > lo) & (a <= hi)].sum())
                   for lo, hi in zip(bounds[:-1], bounds[1:])]
            ax.plot(eq.atoms + (k - 0.5) * 0.03 * xm, agg, "o", ms=4.5,
                    color=color, mec="white", mew=0.5, zorder=5)
            ymax = max(ymax, max(agg), float(p.max()))
        ax.vlines(eq.atoms, 0.0, eq.weights, color=C_ANA, lw=1.8,
                  ls=(0, (3, 2)), label="analytical")
        ax.plot(eq.atoms, eq.weights, "o", ms=6, mec=C_ANA, mfc="none", mew=1.4)
        # put the annotation in the corner away from the tallest atom
        text_right = eq.atoms[np.argmax(eq.weights)] < xm / 2

    rbar_rm = 0.5 * (rm["mean1"] + rm["mean2"])
    ax.text(
        0.97 if text_right else 0.05, 0.93,
        f"$\\bar r$ exact {eq.mean:.4f}\n$\\bar r$ RM {rbar_rm:.4f}",
        transform=ax.transAxes, ha="right" if text_right else "left",
        va="top", fontsize=8, color=INK2,
    )
    ax.set_xlim(x0, xm)
    ax.set_ylim(-0.04 * ymax, 1.12 * ymax)


def make_figure(results, path_base="plots/analytical_vs_rm"):
    ncols = len(results) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.4))
    fig.patch.set_facecolor(SURFACE)

    for ax, (ex, eq, rm) in zip(axes.flat, results):
        panel(ax, eq, rm)
        label = ex["label"]
        if ex["tau"] > 0:
            label = f"$\\tau={ex['tau']:g}$, $\\rho={ex['rho']:g}$\n{eq.check['phase']}"
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.tick_params(labelsize=8, colors=INK2)
        for s in ax.spines.values():
            s.set_color("#d8d7d1")

    for i, P in enumerate((0, 1)):
        axes[i, 0].set_ylabel(f"$P={P}$, $R=1$\ndensity $f(a)$", fontsize=10, color=INK)
        axes[i, 2].set_ylabel("probability mass", fontsize=9, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("action $a$ (risk)", fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="none", ms=5, color=INK2,
                              mec="white", mew=0.5))
    labels.append("RM mass summed per atom")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "Analytical equilibria vs regret matching "
        f"({ACTIONS} shifted actions, {ITERS}$\\times${ACTIONS} RM updates)",
        fontsize=12, color=INK, y=1.06,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    for ext in ("svg", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=180, bbox_inches="tight",
                    facecolor=SURFACE)
    return fig


def print_table(results):
    hdr = (f"{'P':>3} {'tau':>5} {'rho':>4}  {'regime':<28} "
           f"{'rbar exact':>11} {'rbar RM':>9} {'u exact':>9} {'u RM':>9} {'check':>9}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for ex, eq, rm in results:
        regime = (ex["label"].splitlines()[-1] if ex["tau"] == 0
                  else eq.check["phase"])
        check = (eq.check["indifference"] if eq.kind == "continuous"
                 else eq.check["br_gap"])
        print(f"{ex['P']:3g} {ex['tau']:5g} {ex['rho']:4g}  {regime:<28} "
              f"{eq.mean:11.6f} {0.5 * (rm['mean1'] + rm['mean2']):9.6f} "
              f"{eq.u_total:9.6f} {rm['u_total']:9.6f} {check:9.1e}")


def main():
    results = compute_all()
    make_figure(results)
    print_table(results)


if __name__ == "__main__":
    main()
