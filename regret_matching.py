#%%
import numpy as np
from numba import njit, float64
from tqdm.auto import tqdm
import time

from game import gen_actions, reward_matrix


@njit
def normalize(x):
    # if x.sum() == 0:
    #     return np.ones_like(x) / len(x)
    # else:
    #     return x / x.sum()
    x = x + np.finfo(np.float64).eps
    return x / x.sum()


@njit
def relu(x):
    return np.maximum(x, 0)


@njit
def softmax(x):
    p = np.exp(x - np.max(x))
    return p / p.sum()


@njit
def multinomial(prob):
    """
    Sample from multinomial distribution. Inpired by <https://github.com/numba/numba/issues/2539#issuecomment-513982892>

    Parameters
    ----------
    prob: array-like
        Probability distribution

    Returns
    -------
    int
        Sampled index
    """
    return np.searchsorted(np.cumsum(prob), np.random.random(), side="right")


@njit
def _rm(
    reward1,
    reward2,
    total_regret1,
    total_regret2,
    avg_strategy1,
    avg_strategy2,
    iters,
    sm=False,
):

    for _ in range(iters):
        if sm:
            strategy1 = softmax(total_regret1)
            strategy2 = softmax(total_regret2)
        else:
            strategy1 = normalize(relu(total_regret1))
            strategy2 = normalize(relu(total_regret2))

        avg_strategy1 += strategy1
        avg_strategy2 += strategy2

        move1 = multinomial(strategy1)
        move2 = multinomial(strategy2)

        total_regret1 += reward1[:, move2] - reward1[move1, move2]
        total_regret2 += reward2[:, move1] - reward2[move2, move1]


@njit
def _cfr(
    reward1,
    reward2,
    total_regret1,
    total_regret2,
    avg_strategy1,
    avg_strategy2,
    iters,
    sm=False,
):
    for _ in range(iters):
        if sm:
            strategy1 = softmax(total_regret1)
            strategy2 = softmax(total_regret2)
        else:
            strategy1 = normalize(relu(total_regret1))
            strategy2 = normalize(relu(total_regret2))

        avg_strategy1 += strategy1
        avg_strategy2 += strategy2

        total_regret1 += reward1 @ strategy2 - strategy1 @ reward1 @ strategy2

        total_regret2 += reward2 @ strategy1 - strategy2 @ reward2 @ strategy1


def _regret_matching_gen(
    game_parameters,
    actions,
    iters,
    shift=True,
    cfr=False,
    sm=False,
    batch_size=100,
    progress=True,
):
    actions1, actions2 = gen_actions(actions, shift)

    reward1 = reward_matrix(actions1, actions2, **game_parameters)
    reward2 = reward_matrix(actions2, actions1, **game_parameters)
    total_regret1 = np.zeros(actions)
    total_regret2 = np.zeros(actions)

    avg_strategy1 = np.zeros(actions)
    avg_strategy2 = np.zeros(actions)

    if cfr:
        _iter = _cfr
    else:
        # fortran order for better cache locality
        reward1 = np.asfortranarray(reward1)
        reward2 = np.asfortranarray(reward2)
        _iter = _rm
        iters *= actions
        batch_size *= actions

    start = time.time()
    with tqdm(total=iters, disable=not progress) as pbar:
        for it in range(0, iters, batch_size):
            subiters = min(batch_size, iters - it)
            _iter(
                reward1,
                reward2,
                total_regret1,
                total_regret2,
                avg_strategy1,
                avg_strategy2,
                subiters,
                sm,
            )
            callback_start = time.time()
            yield (
                it + subiters,
                (actions1, normalize(avg_strategy1)),
                (actions2, normalize(avg_strategy2)),
                reward1,
                reward2,
                time.time() - start,
            )
            start += time.time() - callback_start
            pbar.update(subiters)


def regret_matching(*args, generator=False, **kwargs):
    """Regret matching

    Parameters
    ----------
    game_parameters : dict
        Game parameters
    actions : int
        Number of actions
    iters : int
        Number of iterations.
        When using "rm" method, iters is multiplied by actions.
    shift : bool, optional
        Shift actions
    cfr : bool, optional
        Use simultaneous CFR instead of RM.
    sm : bool, optional
        Use softmax instead of relu.
    batch_size : int, optional
        Batch size, by default 100.
    generator : bool, optional
        Return generator, by default False.
    progress : bool, optional
        Show progress bar, by default True.
    """
    if generator:
        return _regret_matching_gen(*args, **kwargs)
    for ans in _regret_matching_gen(*args, **kwargs):
        pass
    return ans
