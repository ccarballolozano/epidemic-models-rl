import numpy as np
import gymnasium as gym


def initialize_state_action_values(
    env: gym.Env, type: str = None, random_max: float = 10
) -> np.array:
    if type == "random" or type is None:
        Q = (
            np.random.rand(
                env.observation_space.nvec[0],
                env.observation_space.nvec[1],
                env.action_space.n,
            )
            * random_max
        )
        for m_s in range(Q.shape[0]):
            for m_i in range(Q.shape[1] - m_s, Q.shape[1]):
                Q[m_s, m_i, :] = 0
    elif type == "zeros":
        Q = np.zeros(
            (
                env.observation_space.nvec[0],
                env.observation_space.nvec[1],
                env.action_space.n,
            )
        )
    else:
        raise ValueError(f"Invalid type {type}")
    return Q


def Q_dict_to_array(Q: dict) -> np.array:
    indices = list(Q.keys())
    n_dims = len(indices[0])
    shape = []
    for dim in range(n_dims):
        shape.append(max([index[dim] for index in indices]) + 1)
    Q_arr = np.zeros(shape=shape)
    for k, v in Q.items():
        Q_arr[k] = v
    return Q_arr
