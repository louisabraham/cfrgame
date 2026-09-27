# A game of Competition for Risk

Code for the paper "A game of Competition for Risk". [pdf](https://github.com/louisabraham/cfrgame/blob/master/paper/article.pdf)


## Files

- `experiments.py`: the main file to produce all plots to the `plots/` folder. Launch it to reproduce all our experiments.
- `game.py`: defines the Competition for Risk game
- `multiple_players.py`: analytical solution to the CfR game in the absence frictions and correlations.
- `regret_matching.py`: implements the regret matching algorithm to find correlated equilibria efficiently. Supports discrete and continuous games.
- `nashconv.py`: implements the NashConv metric for exploitability and the novel QuasiNashConv that extends it to continuous games.
- `linear_solver.py`: linear program to solve correlated equilibria and check the diameter of the set of correlated equilibria using our novel method.
- `bivariate_normal.py`: CDF of the bivariate normal distribution, but FASSST (using numba).
- `benchmark.py`: compares our regret-matching solver with Lemke-Howson (Gambit and a numpy tableau), the Mangasarian-Stone / Nash-gap QP (Gurobi, SCIP), a symmetric Fischer-Burmeister NCP, Double Oracle (grid and continuous best responses), multi-start Ipopt, learning dynamics and [nashopt](https://github.com/bemporad/nashopt), for grid sizes 16 to 4096. Quality is the QuasiNashConv with continuous best responses. Run `python benchmark.py run`, then `evaluate`, then `plot`; set `BENCH_TAU=0` for the frictionless game (default 0.03). Results: `plots/benchmark_accuracy_time_tau=<tau>.svg` and `bench/tau=<tau>/RESULTS.md`.
- `baselines.py`: the solvers used by `benchmark.py`.
- `hybrid.py`: Double Oracle followed by Newton on the atom positions and weights (pairs of close atoms are merged during the solve); `newton_atoms` also refines a fictitious-play solution.
- `multiple_oracle.py`: the n-player CfR game with friction (softmax winner among the players that do not fail) and the Multiple Oracle algorithm of Kroupa and Votroubek (arXiv:2109.04178).
- `bench/report/report.pdf`: short report on the approaches and the benchmark results; `python bench/scripts/make_report.py` rebuilds its tables and the PDF. `bench/scripts/run_extra.sh` runs the extra experiments of the report (Double Oracle + Newton, n players).
