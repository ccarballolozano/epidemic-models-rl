import numpy as np

# Q(M_S, M_I, a) is the q value for state (M_S, M_I) and action a. Only values are are when M_S + M_I <= N, where N is the dimension of Q (Q.shape[0])


def get_state_values_mask(n: int):
    mask = np.flip(np.tri(n, n, k=-1), 1)
    return mask


def get_state_action_values_mask(n: int, a: int):
    mask = get_state_values_mask(n)
    mask = np.expand_dims(mask, axis=-1).repeat(a, axis=-1)
    return mask


def get_state_values_masked(V: np.array):
    # assert len(V.shape) == 2, "V must be a 2D tensor"
    # assert V.shape[0] == V.shape[1], "V must be a square matrix"
    mask = get_state_values_mask(V.shape[0])
    return np.ma.array(V, mask=mask)


def get_state_action_values_masked(Q: np.array):
    # assert len(Q.shape) == 3, "Q must be a 3D tensor"
    # assert Q.shape[0] == Q.shape[1], "Q must be a square matrix"
    mask = get_state_action_values_mask(Q.shape[0], Q.shape[2])
    return np.ma.array(Q, mask=mask)


def compute_state_value_error(V_true: np.array, V_approx: np.array):
    """Compute absolute errors for value function (state values only).

    Formula (per valid state): |V_true(s) - V_approx(s)|
    Valid states: those where M_S + M_I <= N.

    Returns:
        (min_error, mean_error, median_error, max_error)
        Index [1] is the mean used for logging as 'mean_absolute_error_V'.
    """
    assert V_true.shape == V_approx.shape, "Shapes of V_true and V_approx must match"

    # Apply the mask to the true and approximated V-values
    mask = get_state_values_mask(V_approx.shape[0])
    V_true_masked = np.ma.array(V_true, mask=mask)
    V_approx_masked = np.ma.array(V_approx, mask=mask)

    # Compute the absolute errors
    errors = np.abs(V_true_masked - V_approx_masked)

    return (
        np.ma.min(errors),
        np.ma.mean(errors),
        np.ma.median(errors),
        np.ma.max(errors),
    )


def compute_state_action_value_error(Q_true: np.array, Q_aprox: np.array):
    """Compute absolute errors for Q-function (state-action values).

    Formula (per valid state-action pair): |Q_true(s,a) - Q_approx(s,a)|
    Valid state-action pairs: those where M_S + M_I <= N and action a is available.

    Returns:
        (min_error, mean_error, median_error, max_error)
        Index [1] is the mean used for logging as 'mean_absolute_error'.
    """
    assert Q_true.shape == Q_aprox.shape, "Shapes of Q_true and Q_aprox must match"

    # Apply the mask to the true and approximated Q-values
    mask = get_state_action_values_mask(Q_aprox.shape[0], Q_aprox.shape[2])
    Q_true_masked = np.ma.array(Q_true, mask=mask)
    Q_aprox_masked = np.ma.array(Q_aprox, mask=mask)

    # Compute the absolute errors
    errors = np.abs(Q_true_masked - Q_aprox_masked)

    return (
        np.ma.min(errors),
        np.ma.mean(errors),
        np.ma.median(errors),
        np.ma.max(errors),
    )


def compute_state_value_relative_error(
    V_true: np.array, V_approx: np.array, subset: str = None
):
    """Compute relative errors for value function (state values only).

    Formula (per valid state): |V_true(s) - V_approx(s)| / |V_true(s)|
    Valid states: those where M_S + M_I <= N and V_true(s) != 0.

    Parameters:
        V_true, V_approx: Value function matrices, shape (N+1, N+1).
        subset: Optional subset filter ("uninfected" or "infected").

    Returns:
        (min_error, mean_error, median_error, max_error)
        Index [1] is the mean used for:
        - 'mean_relative_error_V' (all valid states)
        - 'log_mean_relative_error_V' (log10 of the mean)
    """
    assert V_true.shape == V_approx.shape, "Shapes of V_true and V_approx must match"

    # Apply the mask to the true and approximated V-values
    mask = get_state_values_mask(V_approx.shape[0]).astype(float)
    mask_zeros = (V_true == 0).astype(float)  # TODO: Avoid zeros removal
    mask = np.ma.mask_or(mask, mask_zeros)
    if subset is not None:
        if subset == "uninfected":
            mask_infected = np.ones(V_approx.shape)
            mask_infected[:, 0] = 0
            mask = np.ma.mask_or(mask, np.ma.make_mask(mask_infected))
        elif subset == "infected":
            mask_uninfected = np.zeros(V_approx.shape)
            mask_uninfected[:, 0] = 1
            mask = np.ma.mask_or(mask, np.ma.make_mask(mask_uninfected))
        else:
            raise ValueError(f"Invalid subset {subset}")
    V_true_masked = np.ma.array(V_true, mask=mask)
    V_approx_masked = np.ma.array(V_approx, mask=mask)

    # Compute the relative errors
    relative_errors = np.abs(V_true_masked - V_approx_masked) / np.abs(V_true_masked)

    return (
        np.ma.min(relative_errors),
        np.ma.mean(relative_errors),
        np.ma.median(relative_errors),
        np.ma.max(relative_errors),
    )


def compute_state_action_value_relative_error(Q_true: np.array, Q_aprox: np.array):
    """Compute relative errors for Q-function (state-action values).

    Formula (per valid state-action pair): |Q_true(s,a) - Q_approx(s,a)| / |Q_true(s,a)|
    Valid state-action pairs: those where M_S + M_I <= N and Q_true(s,a) != 0.

    Returns:
        (min_error, mean_error, median_error, max_error)
        Index [1] is the mean used for:
        - 'mean_relative_error' (all valid state-action pairs)
        - 'log_mean_relative_error' (log10 of the mean)
    """
    assert Q_true.shape == Q_aprox.shape, "Shapes of Q_true and Q_aprox must match"

    # Apply the mask to the true and approximated Q-values
    mask = get_state_action_values_mask(Q_aprox.shape[0], Q_aprox.shape[2]).astype(
        float
    )
    mask_zeros = (Q_true == 0).astype(float)  # TODO: Avoid zeros removal
    mask = np.ma.mask_or(mask, mask_zeros)
    Q_true_masked = np.ma.array(Q_true, mask=mask)
    Q_aprox_masked = np.ma.array(Q_aprox, mask=mask)

    # Compute the relative errors
    relative_errors = np.abs(Q_true_masked - Q_aprox_masked) / np.abs(Q_true_masked)

    return (
        np.ma.min(relative_errors),
        np.ma.mean(relative_errors),
        np.ma.median(relative_errors),
        np.ma.max(relative_errors),
    )


def compute_proportion_of_states_with_suboptimal_action(
    Q_true: np.array, Q_approx: np.array
):
    """Compute the proportion of states where the learned policy is suboptimal.

    Formula: (# states with argmax_a Q_approx(s,a) != argmax_a Q_true(s,a)) / (# valid states)
    Valid states: those where M_S + M_I <= N.

    Returns:
        Proportion in [0, 1], logged as 'suboptimal_action_rate'.
    """
    assert Q_true.shape == Q_approx.shape, "Shapes of Q_true and Q_aprox must match"

    optimal_actions_true = np.argmax(Q_true, axis=2)
    optimal_actions_approx = np.argmax(Q_approx, axis=2)
    suboptimal_actions = optimal_actions_true != optimal_actions_approx
    mask = get_state_values_mask(Q_approx.shape[0])
    suboptimal_actions_masked = np.ma.array(suboptimal_actions, mask=mask)

    # Compute the proportion of states with suboptimal actions
    proportion = np.sum(suboptimal_actions_masked) / np.sum(~mask.astype(bool))

    return proportion
