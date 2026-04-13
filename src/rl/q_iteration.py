import argparse
from collections import defaultdict
import os
from datetime import datetime

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def compute_social_optimum_policy(
    size: int,
    encounter_rate: float,
    recovery_rate: float,
    resusceptible_rate: float,
    vaccination_rate: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
    max_iterations: int = 1e6,
) -> list[dict, dict]:

    unif = 1 / (
        (size)
        * (encounter_rate + recovery_rate + resusceptible_rate + vaccination_rate)
    )
    w_V = lambda m_s, m_i, action: unif * vaccination_rate * m_s
    w_I = lambda m_s, m_i, action: unif * encounter_rate * action * m_s * m_i / size
    w_R = lambda m_s, m_i, action: unif * recovery_rate * m_i
    w_S = lambda m_s, m_i, action: unif * resusceptible_rate * (size - m_s - m_i)
    w_hat = (
        lambda m_s, m_i, action: 1
        - w_V(m_s, m_i, action)
        - w_I(m_s, m_i, action)
        - w_R(m_s, m_i, action)
        - w_S(m_s, m_i, action)
    )

    states = [
        (m_s, m_i)
        for m_s in range(size + 1)
        for m_i in range(size + 1)
        if m_s + m_i <= size
    ]
    actions = [0, 1]
    Q = defaultdict(
        lambda: 0, {(m_s, m_i, a): 0 for (m_s, m_i) in states for a in actions}
    )
    state_action_keys = list(Q)
    cost = lambda m_s, m_i, action: (cost_lockdown - action) * (
        m_s / size
    ) + cost_infection * (m_i / size)
    reward = lambda m_s, m_i, action: -(cost_lockdown - action) * (
        m_s / size
    ) - cost_infection * (m_i / size)
    next_expected_value = (
        lambda m_s, m_i, action, Q: w_I(m_s, m_i, action)
        * max([Q[m_s - 1, m_i + 1, a] for a in actions])
        + w_V(m_s, m_i, action) * max([Q[m_s - 1, m_i, a] for a in actions])
        + w_R(m_s, m_i, action) * max([Q[m_s, m_i - 1, a] for a in actions])
        + w_S(m_s, m_i, action) * max([Q[m_s + 1, m_i, a] for a in actions])
        + w_hat(m_s, m_i, action) * max([Q[m_s, m_i, a] for a in actions])
    )
    k = 0
    while True:
        k += 1
        Q_k = Q.copy()
        for m_s, m_i, a in state_action_keys:
            Q[m_s, m_i, a] = reward(
                m_s, m_i, a
            ) + discount_factor * next_expected_value(m_s, m_i, a, Q_k)
        delta = np.max(np.abs(list({s: Q[s] - Q_k[s] for s in Q}.values())))
        if delta < theta:
            logger.info(
                f"Social Optimum computation - Converged after {k} iterations, Delta: {delta}"
            )
            break
        elif k >= max_iterations:
            logger.warning(
                f"Social Optimum computation - Max iterations reached: {max_iterations} without convergence, Delta: {delta}"
            )
            break
        else:
            if k % 1000 == 0:
                logger.debug(
                    f"Social Optimum computation - Iteration: {k}, Delta: {delta}"
                )

    policy = {state: 0 for state in states}
    for state in states:
        m_s, m_i = state
        policy[m_s, m_i] = actions[np.argmax([Q[m_s, m_i, a] for a in actions])]

    Q = {sa: Q[sa] for sa in state_action_keys}
    return policy, Q


def main(args):
    import os

    policy, Q = compute_social_optimum_policy(
        args.size,
        args.encounter_rate,
        args.recovery_rate,
        args.resusceptible_rate,
        args.vaccination_rate,
        args.cost_infection,
        args.cost_lockdown,
        args.discount_factor,
        args.theta,
        args.max_iterations,
    )

    f, ax = plt.subplots()
    for m_s, m_i in policy:
        if policy[m_s, m_i] == 0:
            ax.plot(m_s, m_i, "x", color="red", label=r"$\pi^{opt}=0$")
        else:
            ax.plot(m_s, m_i, "o", color="green", label=r"$\pi^{opt}=1$")
    # ax.set_title("Social Optimum Policy")
    ax.set_xlabel("$M_S$")
    ax.set_ylabel("$M_I$")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    f.savefig(
        os.path.join(
            args.output_dir,
            f"sirs_social_plt_{args.size}_{args.encounter_rate}_{args.recovery_rate}_{args.resusceptible_rate}_{args.vaccination_rate}_{args.cost_infection}_{args.cost_lockdown}.png",
        )
    )
    f.show()
    f, ax = plt.subplots()
    # policy dictionary to numpy array, where keys are (row, column) and value is the array value for that row and column
    Q_0 = np.array(
        [
            [Q.get((m_s, m_i, 0), 0) for m_i in range(args.size + 1)]
            for m_s in range(args.size + 1)
        ]
    )
    Q_1 = np.array(
        [
            [Q.get((m_s, m_i, 1), 0) for m_i in range(args.size + 1)]
            for m_s in range(args.size + 1)
        ]
    )
    Q_diff = Q_1 - Q_0
    # im = ax.imshow(Q_diff.T, cmap="RdBu", norm=None)
    max_abs_value = np.max(np.abs(Q_diff))
    cmap = plt.get_cmap("RdBu")
    cmap.set_bad(color="white")
    im = ax.imshow(Q_diff.T, cmap=cmap, vmin=-max_abs_value, vmax=max_abs_value)
    # show values
    for i in range(args.size + 1):
        for j in range(args.size + 1):
            if not i + j <= args.size:
                continue
            text = ax.text(
                i,
                j,
                f"{Q_diff[i, j]:.1e}",
                ha="center",
                va="center",
                color="black",
            )
    # set ticks
    ax.set_xticks(np.arange(args.size + 1))
    ax.set_yticks(np.arange(args.size + 1))
    ax.invert_yaxis()
    # show colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Q(1) - Q(0)")
    ax.set_xlabel("Susceptible")
    ax.set_ylabel("Infected")
    ax.set_title("Q(1) - Q(0)")
    f.savefig(
        os.path.join(
            args.output_dir,
            f"sirs_social_q_diff_{args.size}_{args.encounter_rate}_{args.recovery_rate}_{args.resusceptible_rate}_{args.vaccination_rate}_{args.cost_infection}_{args.cost_lockdown}.png",
        )
    )
    f.show()

    f, ax = plt.subplots()
    Q_best = np.maximum(Q_0, Q_1)
    max_abs_value = np.max(np.abs(Q_best))
    cmap = plt.get_cmap("RdBu")
    cmap.set_bad(color="white")
    im = ax.imshow(Q_best.T, cmap=cmap, vmin=-max_abs_value, vmax=0)
    # show values
    for i in range(args.size + 1):
        for j in range(args.size + 1):
            if not i + j <= args.size:
                continue
            text = ax.text(
                i,
                j,
                f"{Q_best[i, j]:.1e}",
                ha="center",
                va="center",
                color="black",
            )
    # set ticks
    ax.set_xticks(np.arange(args.size + 1))
    ax.set_yticks(np.arange(args.size + 1))
    ax.invert_yaxis()
    # show colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("max_a Q")
    ax.set_xlabel("Susceptible")
    ax.set_ylabel("Infected")
    ax.set_title("max_a Q")
    f.savefig(
        os.path.join(
            args.output_dir,
            f"sirs_social_q_best_{args.size}_{args.encounter_rate}_{args.recovery_rate}_{args.resusceptible_rate}_{args.vaccination_rate}_{args.cost_infection}_{args.cost_lockdown}.png",
        )
    )
    f.show()

    # Plot probabilities
    unif = 1 / (
        (args.size)
        * (
            args.encounter_rate
            + args.recovery_rate
            + args.resusceptible_rate
            + args.vaccination_rate
        )
    )
    w_V = lambda m_s, m_i, action: unif * args.vaccination_rate * m_s
    w_I = (
        lambda m_s, m_i, action: unif
        * args.encounter_rate
        * action
        * m_s
        * m_i
        / args.size
    )
    w_R = lambda m_s, m_i, action: unif * args.recovery_rate * m_i
    w_S = (
        lambda m_s, m_i, action: unif
        * args.resusceptible_rate
        * (args.size - m_s - m_i)
    )
    w_hat = (
        lambda m_s, m_i, action: 1
        - w_V(m_s, m_i, action)
        - w_I(m_s, m_i, action)
        - w_R(m_s, m_i, action)
        - w_S(m_s, m_i, action)
    )
    w_V_arr = np.zeros((args.size + 1, args.size + 1, 2))
    w_I_arr = np.zeros((args.size + 1, args.size + 1, 2))
    w_R_arr = np.zeros((args.size + 1, args.size + 1, 2))
    w_S_arr = np.zeros((args.size + 1, args.size + 1, 2))
    w_hat_arr = np.zeros((args.size + 1, args.size + 1, 2))

    for m_s in range(args.size + 1):
        for m_i in range(args.size + 1 - m_s):
            for a in [0, 1]:
                w_V_arr[m_s, m_i, a] = w_V(m_s, m_i, a)
                w_I_arr[m_s, m_i, a] = w_I(m_s, m_i, a)
                w_R_arr[m_s, m_i, a] = w_R(m_s, m_i, a)
                w_S_arr[m_s, m_i, a] = w_S(m_s, m_i, a)
                w_hat_arr[m_s, m_i, a] = w_hat(m_s, m_i, a)
    assert np.allclose(w_V_arr[:, :, 0], w_V_arr[:, :, 1])
    assert not np.allclose(w_I_arr[:, :, 0], w_I_arr[:, :, 1])
    assert np.allclose(w_R_arr[:, :, 0], w_R_arr[:, :, 1])
    assert np.allclose(w_S_arr[:, :, 0], w_S_arr[:, :, 1])
    assert not np.allclose(w_hat_arr[:, :, 0], w_hat_arr[:, :, 1])
    # TODO: The sum is not 1, each element must be 1, but only for defined states
    # assert np.allclose(
    #    w_V_arr[:, :, 0]
    #    + w_I_arr[:, :, 0]
    #    + w_R_arr[:, :, 0]
    #    + w_S_arr[:, :, 0]
    #    + w_hat_arr[:, :, 0],
    #    1,
    # )
    # assert np.allclose(
    #    w_V_arr[:, :, 1]
    #    + w_I_arr[:, :, 1]
    #    + w_R_arr[:, :, 1]
    #    + w_S_arr[:, :, 1]
    #    + w_hat_arr[:, :, 1],
    #    1,
    # )

    f, axs = plt.subplots(2, 3)
    arrays = [
        w_V_arr[:, :, 0],
        w_I_arr[:, :, 0],
        w_I_arr[:, :, 1],
        w_R_arr[:, :, 0],
        w_S_arr[:, :, 0],
        w_hat_arr[:, :, 1],
    ]
    titles = ["w_V", "w_I (a=0)", "w_I (a=1)", "w_R", "w_S", "w_hat (a=1)"]

    for i, arr in enumerate(arrays):
        ax = axs[i // axs.shape[1], i % axs.shape[1]]
        cax = ax.imshow(arr.T, cmap="Blues", vmin=0, vmax=1)

        f.colorbar(cax, ax=ax)
        ax.set_xticks(np.arange(args.size + 1))
        ax.set_yticks(np.arange(args.size + 1))
        ax.invert_yaxis()
        ax.set_title(f"{titles[i]}")

    f.show()
    print(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--encounter_rate", type=float, default=0.6)
    parser.add_argument("--recovery_rate", type=float, default=0.4)
    parser.add_argument("--resusceptible_rate", type=float, default=0.2)
    parser.add_argument("--vaccination_rate", type=float, default=0.2)
    parser.add_argument("--cost_infection", type=float, default=6.7)
    parser.add_argument("--cost_lockdown", type=float, default=2)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-6)
    parser.add_argument("--max_iterations", type=int, default=1e6)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
    print(0)
