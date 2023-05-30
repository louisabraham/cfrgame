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