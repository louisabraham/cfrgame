import numpy as np
import scipy
import torch

from game import reward, reward_matrix

from scipy.optimize import linprog


def _gen_constraints(reward1, reward2):
    n_actions1 = len(reward1)
    n_actions2 = len(reward2)
    eqs = []
    action2 = np.arange(n_actions2)
    for action1 in range(n_actions1):
        for action1_alt in range(n_actions1):
            if action1_alt == action1:
                continue
            eq = np.zeros((n_actions1, n_actions2))
            eq[action1, action2] = reward1[action1_alt] - reward1[action1]
            eqs.append(eq.flatten())
    action1 = np.arange(n_actions1)
    for action2 in range(n_actions2):
        for action2_alt in range(n_actions2):
            if action2_alt == action2:
                continue
            eq = np.zeros((n_actions1, n_actions2))
            eq[action1, action2] = reward2[action2_alt] - reward2[action2]
            eqs.append(eq.flatten())

    A_ub = np.stack(eqs)
    b_ub = np.zeros(len(A_ub))
    A_eq = np.ones((1, n_actions1 * n_actions2))
    b_eq = np.ones(1)
    bounds = [(0, 1) for _ in range(n_actions1 * n_actions2)]
    return {"A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq, "bounds": bounds}


def linear_correlated(reward1, reward2, n_iters=10, confidence=0.95):
    """Linear program to
    - find a correlated equilibirum of maximum total utility
    - check the diameter of the set of correlated equilibria

    Parameters
    ----------
    reward1
        Reward matrix for player 1 of shape (n_actions1, n_actions2)
    reward2
        Reward matrix for player 2 of shape (n_actions2, n_actions1)

    Returns
    -------
    eq
        Correlated equilibrium of shape (n_actions1, n_actions2)
    utility
        Total utility of the correlated equilibrium
    diameter
        Upper bound on the diameter of the set of correlated equilibria
        with the given confidence.
    eig_ratio
        Ratio of the second to first singular value of the equilibrium.
        This is a measure of the correlation as `eig_ratio == 0`
        iff the equilibrium is uncorrelated (i.e. a mixed Nash equilibrium).
    """
    n_actions1 = len(reward1)
    n_actions2 = len(reward2)
    constraints = _gen_constraints(reward1, reward2)
    d = np.empty(n_iters)
    for i in range(n_iters):
        c = np.random.randn(n_actions1 * n_actions2)
        res = linprog(c, **constraints)
        assert res.success
        a = res.x @ c
        res = linprog(-c, **constraints)
        assert res.success
        b = res.x @ c
        d[i] = b - a
    e = np.sum(d**2)
    q = scipy.stats.chi2.ppf(1 - confidence, n_iters)
    c = -(reward1 + reward2.T).flatten()
    res = linprog(c, **constraints)
    assert res.success
    flat = res.x
    eq = flat.reshape(n_actions1, n_actions2)
    eig = np.linalg.svd(eq, compute_uv=False)
    utility1 = flat @ reward1.flatten()
    utility2 = flat @ reward2.T.flatten()
    return (eq, (utility1, utility2), min(2, e / q), eig[1] / eig[0])


#%%


if __name__ == "__main__":
    # chicken game <https://en.wikipedia.org/wiki/Correlated_equilibrium#An_example>
    reward1 = np.array([[0.0, 7], [2, 6]])
    reward2 = reward1
    eq, utility, d_max, second = linear_correlated(
        reward1, reward2, n_iters=10, confidence=0.95
    )
    assert sum(utility) == 10.5

# %%
