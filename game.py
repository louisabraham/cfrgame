"""
Utilities to compute the reward of the CoR game.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from bivariate_normal import bvn_cdf, norm_ppf
from numba import njit, float64, vectorize


def joint_failure_probability_slow(a, b, cov):
    """Probability of both players failing at the same time.
    Player 1 takes risk a, player 2 takes risk b.
    cov is the covariance of the two portfolios.
    """
    # cov = 2 * torch.sin(torch.pi / 6 * cov)
    if cov == 0:
        return a * b
    args = np.stack((norm.ppf(a), norm.ppf(b)), axis=-1)
    ans = multivariate_normal.cdf(
        args,
        cov=[[1, cov], [cov, 1]],
        allow_singular=True,
    )
    if isinstance(ans, np.ndarray):
        ans[np.isnan(ans)] = (a * b)[np.isnan(ans)]
    elif np.isnan(ans):
        ans = a * b
    return ans


@njit
def joint_failure_probability(a, b, cov):
    """Probability of both players failing at the same time.
    Player 1 takes risk a, player 2 takes risk b.
    cov is the covariance of the two portfolios.
    """
    if cov == 0:
        return a * b
    if cov == -1:
        return np.maximum(0, a + b - 1)
    if cov == 1:
        return np.minimum(a, b)
    return bvn_cdf(norm_ppf(a), norm_ppf(b), cov)


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@vectorize([float64(float64)])
def sign_strict(x):
    if x > 0:
        return 1
    return 0


@njit
def reward(a, b, corr=0.0, noise=0, R=1, Z=0, P=1):
    """Reward of Player 1

    Inputs
    ------
    a: float or array-like
        Action of player 1
    b: float or array-like
        Action of player 2
    corr: float
        Correlation
    noise: float
        Market noise level, < 0 for strict
    R: float
        Reward
    Z: float
        Null reward
    P: float
        Penalty
    """

    c = joint_failure_probability(a, b, cov=corr)
    left = R * (b - c) - a * P + (1 - a - b + c) * Z
    right = (1 - a) * R - a * P
    if noise < 0:
        th = sign_strict(a - b)
    elif noise == 0:
        th = (np.sign(a - b) + 1) / 2
    else:
        th = sigmoid((a - b) / noise)
    return left + th * (right - left)


def _cast(x):
    if isinstance(x, (int, float)):
        x = np.array([x])
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


def reward_matrix(a, b, **kwargs):
    """Returns a matrix of rewards for all combinations of a and b"""
    a = _cast(a)
    b = _cast(b)
    la = len(a)
    lb = len(b)
    a = a.reshape(la, 1).repeat(lb, axis=1).flatten()
    b = b.reshape(1, lb).repeat(la, axis=0).flatten()
    return reward(a, b, **kwargs).reshape(la, lb)


def gen_actions(n_actions, shift):
    if shift:
        actions1 = np.linspace(0, 1, 2 * n_actions)[::2]
        actions2 = np.linspace(0, 1, 2 * n_actions)[1::2]
    else:
        actions1 = np.linspace(0, 1, n_actions)
        actions2 = np.linspace(0, 1, n_actions)
    return actions1, actions2
