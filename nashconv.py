# use sobol random numbers
from scipy import stats
from tqdm.autonotebook import tqdm

from game import reward, reward_matrix


def nashconv(matrix1, matrix2, prob1, prob2):
    """Determine the total exploitability

    Parameters
    ----------

    matrix1
        Matrix of rewards for player 1 with shape (n, m)
    matrix2
        Matrix of rewards for player 2 with shape (m, n)
    prob1
        Probability distribution of player 1 with shape (n,)
    prob2
        Probability distribution of player 2 with shape (m,)
    """
    out1 = matrix1 @ prob2
    out2 = matrix2 @ prob1
    return out1.max() - prob1 @ out1 + out2.max() - prob2 @ out2


def quasinashconv(
    game_parameters,
    actions1,
    actions2,
    prob1,
    prob2,
    log_num_points,
    progress=True,
    matrix1=None,
    matrix2=None,
):
    if matrix1 is None:
        matrix1 = reward_matrix(actions1, actions2, **game_parameters)
    if matrix2 is None:
        matrix2 = reward_matrix(actions2, actions1, **game_parameters)
    avg1 = prob1 @ matrix1 @ prob2
    avg2 = prob2 @ matrix2 @ prob1
    rand_actions1 = stats.qmc.Sobol(d=1).random_base2(log_num_points)
    rand_actions2 = stats.qmc.Sobol(d=1).random_base2(log_num_points)
    max1 = max(
        [
            reward(action1, actions2, **game_parameters) @ prob2
            for action1 in tqdm(rand_actions1, disable=not progress)
        ]
    )
    max2 = max(
        [
            reward(action2, actions1, **game_parameters) @ prob1
            for action2 in tqdm(rand_actions2, disable=not progress)
        ]
    )
    return max1 - avg1 + max2 - avg2
