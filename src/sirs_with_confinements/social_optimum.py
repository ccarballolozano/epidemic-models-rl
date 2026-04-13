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
    V = defaultdict(lambda: 0, {state: 0 for state in states})

    cost = lambda m_s, m_i, action: (cost_lockdown - action) * (
        m_s / size
    ) + cost_infection * (m_i / size)
    next_expected_value = (
        lambda m_s, m_i, action, V: w_I(m_s, m_i, action) * V[m_s - 1, m_i + 1]
        + w_V(m_s, m_i, action) * V[m_s - 1, m_i]
        + w_R(m_s, m_i, action) * V[m_s, m_i - 1]
        + w_S(m_s, m_i, action) * V[m_s + 1, m_i]
        + w_hat(m_s, m_i, action) * V[m_s, m_i]
    )
    k = 0
    while True:
        k += 1
        V_k = V.copy()
        for state in states:
            m_s, m_i = state
            V[m_s, m_i] = min(
                [
                    cost(m_s, m_i, action)
                    + discount_factor * next_expected_value(m_s, m_i, action, V_k)
                    for action in [0, 1]
                ]
            )
        delta = np.max(np.abs(list({s: V[s] - V_k[s] for s in V}.values())))
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
        policy[m_s, m_i] = np.argmin(
            [
                cost(m_s, m_i, action)
                + discount_factor * next_expected_value(m_s, m_i, action, V)
                for action in [0, 1]
            ]
        )

    V = {state: V[state] for state in states}
    policy = {state: policy[state] for state in states}
    return policy, V


def main(args):
    import os

    policy, V = compute_social_optimum_policy(
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
            ax.plot(m_s, m_i, "x", color="red", label="confinement")
        else:
            ax.plot(m_s, m_i, "o", color="green", label="max exposure")
    ax.set_title("Social Optimum Policy")
    ax.set_xlabel("Susceptible")
    ax.set_ylabel("Infected")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.savefig(
        os.path.join(
            f"sirs_social_plt_{args.size}_{args.encounter_rate}_{args.recovery_rate}_{args.resusceptible_rate}_{args.vaccination_rate}_{args.cost_infection}_{args.cost_lockdown}.png",
        )
    )
    plt.show(block=True)


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
    args = parser.parse_args()
    main(args)
    print(0)
