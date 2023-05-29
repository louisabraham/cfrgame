"""
Plot all figures from the paper
"""
import json
from pathlib import Path
import time
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from cycler import cycler
import scipy
from tqdm_joblib import tqdm_joblib
import numpy as np
import joblib
from tqdm import tqdm

from game import reward, reward_matrix, gen_actions
from linear_solver import linear_correlated
from multiple_players import find_w, find_cutoff, solution, log_efficiency
from nashconv import nashconv, quasinashconv
from regret_matching import regret_matching, softmax


monochrome = (
    cycler("color", ["k"])
    * cycler("marker", ["", ".", "o"])
    * cycler("linestyle", ["-", "--", ":", "-."])
)
plt.rc("axes", prop_cycle=monochrome)
# plt.rc('figure', autolayout=True)


def plot_reward():
    plt.clf()
    x = np.linspace(0, 1, 10000)
    b = 0.2
    plt.plot(x, reward(x, b))
    plt.title("$u_1(r_1, r_2=0.2)$ with $R = P = 1$")
    plt.xlabel("$r_1$")
    plt.savefig("plots/reward.svg")


sqrt = np.sqrt


def opt(x, c):
    "Nash equilibrium"
    k = sqrt(c * c + 2 * c + 2)
    h = 1 - sqrt((k - 1) / (k + 1))
    return (x < h) * (k - 1) / (1 - x) ** 3


def plot_nash(c):
    plt.clf()
    x = np.linspace(0, 1, 10000)[1:-1]
    plt.plot(x, opt(x, c))
    plt.title(f"Nash equilibrium with $P={c}$")
    plt.xlabel("$r$")
    plt.ylabel("$f(r)$")
    plt.savefig(f"plots/nash-{c}.svg")


def cutoff(c):
    k = sqrt(c * c + 2 * c + 2)
    return 1 - sqrt((k - 1) / (k + 1))


def plot_cutoff():
    plt.clf()
    c = np.linspace(0, 10, 1000)
    plt.plot(c, cutoff(c))
    plt.title("Cutoff value ${r_{max}}$")
    plt.xlabel("$C$")
    plt.ylabel("${r_{max}}$")
    plt.savefig("plots/cutoff.svg")


def plot_response():
    plt.clf()
    c = 1
    x = np.linspace(0, 1, 10000)[1:-1]
    p = opt(x, c)
    p /= p.sum()
    plt.plot(x, [(reward(y, x, R=1, P=1) * p).sum() for y in x])
    plt.title("Reward $u_2$ when Player 1 plays according to the Nash equilibrium")
    plt.xlabel("$r_2$")
    plt.ylabel("$u_2$")
    plt.savefig("plots/response.svg")


def plot_solutions_multiple(C):
    plt.clf()
    x = np.linspace(0, 1, 10000)
    for n in range(2, 6):
        plt.plot(x, solution(n, C, x), label=f"$n={n}$")
    plt.title(f"Solution for $P={C}$")
    plt.xlabel("$r$")
    plt.ylabel("$f(r)$")
    plt.ylim(0, min(plt.ylim()[1], 10))
    plt.legend()
    plt.savefig(f"plots/solution-multiple-{C}.svg")


def plt_asymptotic(C):
    plt.clf()
    all_n = np.unique(np.array(10 ** np.linspace(1, 4, 1000)).astype(int))
    plt.title(f"Asymptotic behavior for $P = {C}$")
    all_C = [C for n in all_n]
    all_w = [find_w(n, C) for n, C in zip(all_n, all_C)]
    all_avg = [np.exp(w / (n - 1)) for n, w in zip(all_n, all_w)]
    all_cutoff = [find_cutoff(n, C) for n, C, w in zip(all_n, all_C, all_w)]
    all_eff = [np.exp(w) * n for n, w in zip(all_n, all_w)]
    plt.plot(all_n, all_cutoff, label=r"${r_{max}}$")
    plt.plot(all_n, all_avg, label=r"$\bar r$")
    plt.plot(all_n, all_eff, label=r"$E$")
    plt.legend()
    plt.xlabel("number of players $n$")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-10, 1)
    plt.savefig(f"plots/efficiency-{C}.svg")


def cutoff_asymptotic():
    plt.clf()
    n = 10000
    all_C = 10 ** np.linspace(-3, 3, 1000)
    all_cutoff = [find_cutoff(n, C) for C in all_C]
    plt.xscale("log")
    plt.plot(all_C, all_cutoff)
    plt.title(f"Cutoff $r_{{max}}$ with $n = {n}$")
    plt.xlabel("$C$")
    plt.ylabel("${r_{max}}$")
    plt.savefig("plots/cutoff-asymptotic.svg")


def limit_constant_c():
    plt.clf()
    C = 1
    n = np.unique(np.array(10 ** np.linspace(1, 4, 1000)).astype(int))
    all_w = np.array([find_w(n, 1, tol=1e-9) for n in n])
    plt.plot(n, all_w + (n - 1) * (np.log(C) + np.log(n)))
    plt.xscale("log")
    plt.savefig("plots/w-n.svg")

    plt.clf()
    all_avg = np.array([np.exp(w / (n - 1)) for n, w in zip(n, all_w)])
    plt.plot(n, all_avg * n * C - 1)
    plt.xscale("log")
    plt.savefig("plots/avg-n.svg")

    plt.clf()
    n = 10000
    C = 10 ** np.linspace(-3, 3, 10000)
    all_w = np.array([find_w(n, C, tol=1e-9) for C in C])
    plt.plot(C, all_w + (n - 1) * (np.log(C) + np.log(n)))
    plt.xscale("log")
    plt.savefig("plots/w-C.svg")

    plt.clf()
    all_avg = np.array([np.exp(w / (n - 1)) for w in all_w])
    plt.plot(C, all_avg * n * C - 1)
    plt.xscale("log")
    plt.savefig("plots/avg-C.svg")


def efficiency():
    plt.clf()
    all_n = np.unique(np.array(10 ** np.linspace(0.4, 3, 30)).astype(int))
    for e in [0, 0.5, 0.9, 1, 1.1, 1.5]:
        all_C = [1 / n**e for n in all_n]
        all_eff = np.array([log_efficiency(n, C) for n, C in zip(all_n, all_C)])
        plt.plot(all_n, np.exp(all_eff), label=f"$e={e}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-2, 1)
    plt.title("Efficiency with $P=n^{-e}$ for various values of $e$")
    plt.ylabel("Efficiency")
    plt.xlabel("Number of players $n$")
    plt.legend()
    plt.savefig("plots/efficiency.svg")


def linear_experiment(corr, noise, P, n_actions, shift):
    start = time.time()
    game_parameters = dict(R=1, Z=0, P=P, corr=corr, noise=noise)
    actions1, actions2 = gen_actions(n_actions, shift)
    reward1 = reward_matrix(actions1, actions2, **game_parameters)
    reward2 = reward_matrix(actions2, actions1, **game_parameters)
    for _ in range(10):
        try:
            eq, utility, d_max, eigen_ratio = linear_correlated(
                reward1, reward2, n_iters=5, confidence=0.95
            )
        except Exception:
            continue
        else:
            break
    else:
        eq, utility, d_max, eigen_ratio = np.nan, np.nan, np.nan, np.nan

    return corr, noise, P, n_actions, shift, d_max, eigen_ratio, time.time() - start


def make_linear_experiments():
    exps = []
    for corr in -1, -0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9, 1:
        for noise in -1, 0, 1e-4, 1e-3, 1e-2:
            for P in 0, 0.1, 0.5, 1, 2, 10:
                for n_actions in 25, 50:
                    for shift in False, True:
                        exps.append(
                            joblib.delayed(linear_experiment)(
                                corr, noise, P, n_actions, shift
                            )
                        )

    # parallelize calls to linear_experiment to use joblib
    with tqdm_joblib(total=len(exps)):
        results = joblib.Parallel(n_jobs=-2)(exps)
    with open("plots/linear-results.json", "w") as f:
        json.dump(results, f)


def linear_exp_plots():
    if not Path("plots/linear-results.json").exists():
        make_linear_experiments()
    with open("plots/linear-results.json") as f:
        results = json.load(f)
    print(
        f"Did {len(results)} linear experiments in",
        time.strftime("%H:%M:%S", time.gmtime(sum(r[-1] for r in results))),
    )
    for s in False, True:
        # collect values of corr, noise, P, n_actions and shift in sorted lists
        corr = sorted(list(set([r[0] for r in results])))
        noise = sorted(list(set([r[1] for r in results])))
        P = sorted(list(set([r[2] for r in results])))
        n_actions = sorted(list(set([r[3] for r in results])))
        results_dict = {tuple(it[:5]): it[5:] for it in results}

        d_max_grid = np.array(
            [
                [results_dict[(c, n, p, a, s)][0] for p in P for n in noise]
                for a in n_actions
                for c in corr
            ]
        )
        eigen_ratio_grid = np.array(
            [
                [results_dict[(c, n, p, a, s)][1] for p in P for n in noise]
                for a in n_actions
                for c in corr
            ]
        )
        for img, title, filename in [
            (d_max_grid, r"$d_{max}$", f"linear-dmax-{'shift' if s else 'no-shift'}"),
            (
                eigen_ratio_grid,
                "$\lambda$",
                f"linear-lambda-{'shift' if s else 'no-shift'}",
            ),
        ]:
            plt.clf()
            plt.imshow(img, cmap="gray_r", norm=LogNorm(vmin=1e-15, vmax=1))
            x_labels_n = [
                f"{n}" for p in P for n in ["strict", 0, "1e-4", "1e-3", "1e-2"]
            ]
            x_labels_p = [
                f"{p}" for p in P for n in ["strict", 0, "1e-4", "1e-3", "1e-2"]
            ]
            plt.xticks(
                range(len(x_labels_n)),
                x_labels_n,
                rotation=90,
                va="center",
                ha="center",
            )
            for tick in plt.xticks()[1]:
                tick.set_y(tick.get_position()[1] - 0.07)
            for i, label in enumerate(x_labels_p):
                if i % len(noise) == (len(noise) + 1) // 2:
                    plt.text(
                        i,
                        len(n_actions) * len(corr) + 3.5,
                        label,
                        ha="center",
                        va="center",
                        rotation=90,
                    )
            y_labels_c = [f"{c}" for a in n_actions for c in corr]
            y_labels_a = [f"{a}" for a in n_actions for c in corr]
            plt.yticks(range(len(y_labels_c)), y_labels_c, va="center", ha="center")
            for tick in plt.yticks()[1]:
                tick.set_x(tick.get_position()[0] - 0.03)
            for i, label in enumerate(y_labels_a):
                if i % len(corr) == (len(corr) + 1) // 2:
                    plt.text(-4, i, label, ha="center", va="center")
            plt.title(title)
            plt.xlabel(r"$C$, $\tau$", labelpad=25)
            plt.ylabel(r"actions, $\rho$", labelpad=25)
            plt.colorbar()
            # scatter plot crosses where there is nan
            plt.scatter(
                np.where(np.isnan(img))[1],
                np.where(np.isnan(img))[0],
                marker="x",
                color="k",
            )
            plt.savefig(f"plots/{filename}.svg")


def compute_rm(cfr, sm, shift, log_actions, log_quasi):
    game_params = dict(corr=0.0, noise=0, R=1, Z=0, P=1)
    its = []
    rewards = []
    nashconv_values = []
    quasinashconv_values = []
    computation_time = []
    for it, (a1, p1), (a2, p2), r1, r2, t in regret_matching(
        game_params,
        actions=2**log_actions,
        iters=10000,
        cfr=cfr,
        sm=sm,
        shift=shift,
        generator=True,
        progress=False,
    ):
        its.append(it)
        rewards.append((p1 @ r1 @ p2, p2 @ r2 @ p1))
        nashconv_values.append(nashconv(r1, r2, p1, p2))
        quasinashconv_values.append(
            quasinashconv(
                game_params,
                a1,
                a2,
                p1,
                p2,
                log_num_points=log_quasi,
                progress=False,
                matrix1=r1,
                matrix2=r2,
            )
        )
        computation_time.append(t)
    return {
        "cfr": cfr,
        "sm": sm,
        "shift": shift,
        "log_actions": log_actions,
        "log_quasi": log_quasi,
        "its": its,
        "rewards": rewards,
        "nashconv_values": nashconv_values,
        "quasinashconv_values": quasinashconv_values,
        "computation_time": computation_time,
    }


def compute_all_rm():
    exps = []
    for cfr in False, True:
        for sm in False, True:
            for shift in False, True:
                for log_actions in range(7, 12 + cfr):
                    log_quasi = log_actions + 3
                    # delayed joblib
                    exps.append(
                        joblib.delayed(compute_rm)(
                            cfr, sm, shift, log_actions, log_quasi
                        )
                    )
    with tqdm_joblib(total=len(exps)):
        results = joblib.Parallel(n_jobs=-2)(exps)
    with open("plots/regret_matching.json", "w") as f:
        json.dump(results, f)


# %%
def regret_matching_plots():
    if not Path("plots/regret_matching.json").exists():
        compute_all_rm()
    with open("plots/regret_matching.json") as f:
        results = json.load(f)
    results_dict = {
        (d["cfr"], d["sm"], d["shift"], d["log_actions"]): d for d in results
    }
    log_actions = max(d["log_actions"] for d in results if not d["cfr"])
    for name, ylabel, file in [
        ("nashconv_values", "NashConv", "nashconv"),
        ("quasinashconv_values", "QuasiNashConv", "quasinashconv"),
    ]:
        plt.clf()
        for cfr in False, True:
            for sm in False, True:
                for shift in False, True:
                    label = "CFR" if cfr else "RM"
                    if sm:
                        label += ", softmax"
                    if shift:
                        label += ", shift"
                    label += f", $a=2^{{{log_actions + cfr}}}$"
                    d = results_dict[cfr, sm, shift, log_actions + cfr]
                    plt.plot(
                        d["computation_time"],
                        d[name],
                        label=label,
                        lw=0.5,
                        markersize=1,
                    )
        legend = plt.legend(loc="upper right", prop={"size": 7})
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 0.6))
        plt.xlabel("time (s)")
        plt.xscale("log")
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.savefig(f"plots/regret_matching_methods_{file}.svg")

    cfr = False
    sm = False
    shift = False
    for name, ylabel, file in [
        ("nashconv_values", "NashConv", "nashconv"),
        ("quasinashconv_values", "QuasiNashConv", "quasinashconv"),
    ]:
        plt.clf()
        for log_actions in set(d["log_actions"] for d in results):
            try:
                d = results_dict[cfr, sm, shift, log_actions]
                plt.plot(
                    np.array(d["its"]) / 2**log_actions,
                    d[name],
                    label=f"$2^{{{log_actions}}}$ actions",
                )
            except KeyError:
                pass
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.savefig(f"plots/regret_matching_sizes_{file}.svg")


# plot average r, efficiency (utility for C=0), rmax

# %%


def compute_equilibria(corr, noise, C):
    game_params = dict(corr=corr, noise=noise, R=1, Z=0, P=C)
    _, (a1, p1), (a2, p2), m1, m2, _ = regret_matching(
        game_params, actions=500, iters=2000, progress=False
    )

    u1 = p1 @ m1 @ p2
    u2 = p2 @ m2 @ p1

    nc1 = nashconv(m1, m2, p1, p2)
    nc2 = nashconv(m2, m1, p2, p1)

    avg_r1 = p1 @ a1
    avg_r2 = p2 @ a2

    def softargmax(a, p):
        return a @ softmax(1000 * p)

    rmax1 = softargmax(a1, p1)
    rmax2 = softargmax(a2, p2)

    return {
        "corr": corr,
        "noise": noise,
        "C": C,
        "nc1": nc1,
        "nc2": nc2,
        "u1": u1,
        "u2": u2,
        "avg_r1": avg_r1,
        "avg_r2": avg_r2,
        "rmax1": rmax1,
        "rmax2": rmax2,
    }


def compute_all_eq():
    exps = []
    for corr in np.linspace(-1, 1, 21):
        for noise in [-1, 0] + list(10 ** np.linspace(-3, -1, 9)):
            for C in [0] + list(10 ** np.linspace(-2, 1, 7)):
                exps.append(joblib.delayed(compute_equilibria)(corr, noise, C))
    with tqdm_joblib(total=len(exps)):
        results = joblib.Parallel(n_jobs=-2)(exps)
    with open("plots/equilibria.json", "w") as f:
        json.dump(results, f)


def equilibria_img():
    if not Path("plots/equilibria.json").exists():
        compute_all_eq()
    with open("plots/equilibria.json") as f:
        results = json.load(f)
    results_dict = {(d["corr"], d["noise"], d["C"]): d for d in results}
    all_C = sorted(set(d["C"] for d in results))
    all_noise = sorted(set(d["noise"] for d in results))[2:]
    all_corr = sorted(set(d["corr"] for d in results))

    for file, name, f in [
        ("rbar", r"$\bar r$", lambda d: (d["avg_r1"] + d["avg_r2"]) / 2),
        # ("rmax", r"$r_{max}$", lambda d: (d["rmax1"] + d["rmax2"]) / 2),
        ("utility", r"$u$", lambda d: d["u1"] + d["u2"]),
        (
            "nashconv",
            r"$\mathcal{N}$",
            lambda d: (d["nc1"] + d["nc2"]) / (d["u1"] + d["u2"]),
        ),
    ]:
        vmin = None
        vmax = None
        for C in [0, 0.1, 0.31622776601683794, 1, 10]:
            plt.clf()
            grid = [
                [f(results_dict[c, noise, C]) for c in all_corr] for noise in all_noise
            ]
            interp = scipy.interpolate.RegularGridInterpolator(
                (all_noise, all_corr), grid, method="linear"
            )
            smooth_corr = np.linspace(-1, 1, 50)
            smooth_noise = np.logspace(-3, -1, 50)
            smooth_grid = [
                [interp([noise, corr])[0] for corr in smooth_corr]
                for noise in smooth_noise
            ]
            # black and white, black is best
            # plt.pcolormesh(all_corr, all_noise, grid, shading="gouraud", cmap="Greys")
            plt.pcolormesh(
                smooth_corr,
                smooth_noise,
                smooth_grid,
                shading="gouraud",
                rasterized=True,
                cmap="Greys",
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar()
            plt.xlabel(r"$\rho$")
            plt.yscale("log")
            plt.ylabel(r"$\tau$")
            plt.title(f"{name} for $P={C}$")
            plt.savefig(f"plots/equilibria_{file}_grid_C={C}.svg")

        # now do the same for C and noise
        for corr in [-1, 0, 1]:
            plt.clf()
            grid = [
                [f(results_dict[corr, noise, C]) for C in all_C] for noise in all_noise
            ]
            interp = scipy.interpolate.RegularGridInterpolator(
                (all_noise, all_C), grid, method="linear"
            )
            smooth_C = np.logspace(-2, 1, 50)
            smooth_noise = np.logspace(-3, -1, 50)
            smooth_grid = [
                [interp([noise, C])[0] for C in smooth_C] for noise in smooth_noise
            ]
            # black and white, black is best
            # plt.pcolormesh(all_C, all_noise, grid, shading="gouraud", cmap="Greys")
            plt.pcolormesh(
                smooth_C,
                smooth_noise,
                smooth_grid,
                shading="gouraud",
                rasterized=True,
                cmap="Greys",
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar()
            plt.xscale("log")
            plt.xlabel("$C$")
            plt.yscale("log")
            plt.ylabel(r"$\tau$")
            plt.title(f"{name} for $\\rho={corr}$")
            plt.savefig(f"plots/equilibria_{file}_grid_corr={corr}.svg")

        plt.clf()
        # now do the same for C and corr
        noise = 0
        grid = [[f(results_dict[corr, noise, C]) for corr in all_corr] for C in all_C]
        interp = scipy.interpolate.RegularGridInterpolator(
            (all_C, all_corr), grid, method="linear"
        )
        smooth_C = np.logspace(-2, 1, 50)
        smooth_corr = np.linspace(-1, 1, 50)
        smooth_grid = [[interp([C, corr])[0] for corr in smooth_corr] for C in smooth_C]
        # black and white, black is best
        # plt.pcolormesh(all_C, all_corr, grid, shading="gouraud", cmap="Greys")
        plt.pcolormesh(
            smooth_corr,
            smooth_C,
            smooth_grid,
            shading="gouraud",
            rasterized=True,
            cmap="Greys",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar()
        plt.xlabel(r"$\rho$")
        plt.ylabel("$C$")
        plt.yscale("log")
        plt.title(f"{name} for $\\tau={noise}$")
        plt.savefig(f"plots/equilibria_{file}_grid_noise={noise}.svg")

    plt.clf()


def equilibria_plots():
    if not Path("plots/equilibria.json").exists():
        compute_all_eq()
    with open("plots/equilibria.json") as f:
        results = json.load(f)
    for d in results:
        d["rbar"] = (d["avg_r1"] + d["avg_r2"]) / 2
        d["u"] = d["u1"] + d["u2"]

    results_dict = {(d["corr"], d["noise"], d["C"]): d for d in results}
    all_C = sorted(set(d["C"] for d in results))
    all_noise = sorted(set(d["noise"] for d in results))[2:]
    all_corr = sorted(set(d["corr"] for d in results))

    plt.clf()
    corr = 0
    for C in all_C:
        rbar = [results_dict[corr, noise, C]["rbar"] for noise in all_noise]
        u = [results_dict[corr, noise, C]["u"] for noise in all_noise]
        label = f"$\log_{{10}} C={np.log10(C)}$"
        plt.plot(rbar, u, label=label)

    plt.xlabel(r"$\bar r$")
    plt.ylabel(r"$u$")
    legend = plt.legend(loc="lower right", prop={"size": 7})
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.6))
    plt.title("Equilibrium for different values of $\\tau$ at $\\rho=0$")
    plt.savefig("plots/equilibria_rbar_u_tau.svg")

    plt.clf()
    for noise in all_noise:
        rbar = [results_dict[corr, noise, C]["rbar"] for C in all_C]
        u = [results_dict[corr, noise, C]["u"] for C in all_C]
        label = f"$\\log_{{10}}\\tau={np.log10(noise)}$"
        plt.plot(rbar, u, label=label)

    plt.xlabel(r"$\bar r$")
    plt.ylabel(r"$u$")
    legend = plt.legend(loc="lower right", prop={"size": 7})
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.6))
    plt.title("Equilibrium for different values of $C$ at $\\rho=0$")
    plt.savefig("plots/equilibria_rbar_u_C.svg")

    for name in ["rbar", "u"]:
        for C in [0, 1]:
            plt.clf()
            for noise in all_noise:
                u = [results_dict[corr, noise, C][name] for corr in all_corr]
                label = f"$\\log_{{10}}\\tau={np.log10(noise)}$"
                plt.plot(all_corr, u, label=label)
            legend = plt.legend(loc="lower left", prop={"size": 7})
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1, 1, 1, 0.6))
            plt.xlabel(r"$\rho$")
            plt.ylabel(r"$u$" if name == "u" else r"$\bar r$")
            plt.title(f"Equilibrium for different values of $\\tau$ at $P={C}$")
            plt.savefig(f"plots/equilibria_{name}_corr_C={C}.svg")
            plt.show()

    plt.clf()
    noise = 0
    for C in all_C:
        u = [results_dict[corr, noise, C]["u"] for corr in all_corr]
        label = f"$\\log_{{10}}C={np.log10(C)}$"
        plt.plot(all_corr, u, label=label)
    legend = plt.legend(loc="lower left", prop={"size": 7})
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.6))
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$u$")
    plt.title("Equilibrium for different values of $C$ at $\\tau=0$")
    plt.savefig("plots/equilibria_u_corr_C.svg")
    plt.clf()


equilibria_plots()


# %%
def main():
    with tqdm() as pbar:
        real_savefig = plt.savefig

        def fake_savefig(*args, **kwargs):
            pbar.update(1)
            return real_savefig(*args, **kwargs)

        try:
            plt.savefig = fake_savefig

            plot_reward()
            for c in [1, 0.5, 0, 10, 100]:
                plot_nash(c)
            plot_cutoff()
            plot_response()
            for C in 0, 0.1, 1, 2:
                plot_solutions_multiple(C)
            # for C in 0, 0.1, 1, 2:
            #     plt_asymptotic(C)
            cutoff_asymptotic()
            efficiency()
            linear_exp_plots()
            regret_matching_plots()
            equilibria_plots()

        finally:
            plt.savefig = real_savefig


if __name__ == "__main__":
    main()
