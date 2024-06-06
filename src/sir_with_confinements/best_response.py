import argparse
from collections import defaultdict

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def compute_best_response_policy(
    N: int,
    encounter_rate: float,
    recovery_rate: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
    policy: dict,
) -> list[dict, dict]:

    # Note that N is the total number of players, including the player for which we are computing the best response
    unif = 1 / ((N) * (encounter_rate + recovery_rate))
    N = N - 1
    states = [
        (x, m_s, m_i)
        for x in ["S", "I"]
        for m_s in range(N + 1)
        for m_i in range(N + 1)
        if m_s + m_i <= N
    ]
    # When Player i is susceptible
    p_I = (
        lambda m_s, m_i, action: unif * encounter_rate * action * (m_i / N)
    )  # Player i infected
    q_I = (
        lambda m_s, m_i, action: unif
        * m_s
        * encounter_rate
        * policy["S", m_s, m_i]
        * (m_i / N)
    )  # Another player infected
    q_R = (
        lambda m_s, m_i, action: unif * m_i * recovery_rate
    )  # Another player recovered
    p_S_hat = (
        lambda m_s, m_i, action: 1
        - p_I(m_s, m_i, action)
        - q_I(m_s, m_i, action)
        - q_R(m_s, m_i, action)
    )  # No changes in state

    # When Player i is infected
    p_R_ = lambda m_s, m_i, action: unif * recovery_rate  # Player i recovered

    def q_I_(m_s, m_i, action):
        # Another player gets infected
        if m_s >= 1:
            return unif * m_s * encounter_rate * policy["S", m_s, m_i] * ((m_i + 1) / N)
        else:
            return 0

    # q_R_ = lambda m_s, m_i, action: m_i * recovery_rate  # Another player recovered (as q_R)
    p_I_hat = (
        lambda m_s, m_i, action: 1
        - p_R_(m_s, m_i, action)
        - q_I_(m_s, m_i, action)
        - q_R(m_s, m_i, action)
    )

    V = defaultdict(lambda: 0, {state: 0 for state in states})

    cost = lambda x, action: (cost_lockdown - action) * (x == "S") + cost_infection * (
        x == "I"
    )
    next_expected_value_s = (
        lambda m_s, m_i, action, V: p_I(m_s, m_i, action) * V["I", m_s, m_i]
        + q_I(m_s, m_i, action) * V["S", m_s - 1, m_i + 1]
        + q_R(m_s, m_i, action) * V["S", m_s, m_i - 1]
        + p_S_hat(m_s, m_i, action) * V["S", m_s, m_i]
    )
    next_expected_value_i = (
        lambda m_s, m_i, action, V: q_I_(m_s, m_i, action) * V["I", m_s - 1, m_i + 1]
        + q_R(m_s, m_i, action) * V["I", m_s, m_i - 1]
        + p_I_hat(m_s, m_i, action) * V["I", m_s, m_i]
    )
    k = 0
    while True:
        k += 1
        V_k = V.copy()
        for state in states:
            x, m_s, m_i = state
            if x == "S":
                V[x, m_s, m_i] = np.min(
                    [
                        cost(x, action)
                        + discount_factor * next_expected_value_s(m_s, m_i, action, V_k)
                        for action in [0, 1]
                    ]
                )
            elif x == "I":
                V[x, m_s, m_i] = np.min(
                    [
                        cost(x, action)
                        + discount_factor * next_expected_value_i(m_s, m_i, action, V_k)
                        for action in [0, 1]
                    ]
                )
        delta = np.max(np.abs(list({s: V[s] - V_k[s] for s in V}.values())))
        if delta < theta:
            logger.info(
                f"Best Response computation - Converged after {k} iterations, Delta: {delta}"
            )
            break
        else:
            if k % 1000 == 0:
                logger.debug(
                    f"Best Response computation - Iteration: {k}, Delta: {delta}"
                )

    policy_br = {state: 0 for state in states}
    for state in states:
        x, m_s, m_i = state
        if x == "S":
            policy_br[x, m_s, m_i] = np.argmin(
                [
                    cost(x, action)
                    + discount_factor * next_expected_value_s(m_s, m_i, action, V)
                    for action in [0, 1]
                ]
            )
        if x == "I":
            policy_br[x, m_s, m_i] = np.argmin(
                [
                    cost(x, action)
                    + discount_factor * next_expected_value_i(m_s, m_i, action, V)
                    for action in [0, 1]
                ]
            )

    V = {state: V[state] for state in states}
    policy_br = {state: policy_br[state] for state in states}
    return policy_br, V


def main(args):
    encounter_rate = args.encounter_rate
    recovery_rate = args.recovery_rate
    if args.probs_total:
        encounter_rate = encounter_rate / args.N
        recovery_rate = recovery_rate / args.N
    policy, V = compute_best_response_policy(
        args.N,
        encounter_rate,
        recovery_rate,
        args.cost_infection,
        args.cost_lockdown,
        args.discount_factor,
        args.theta,
    )
    logger.info(f"Policy: {policy}")

    f, axs = plt.subplots(1, 1)
    for x, m_s, m_i in policy:
        if policy["S", m_s, m_i] == 0:
            axs.plot(m_s, m_i, "x", color="red", label="confinement")
        else:
            axs.plot(m_s, m_i, "o", color="green", label="max exposure")
    axs.set_title("Nash equilibrium policy")
    axs.set_xlabel("Susceptible")
    axs.set_ylabel("Infected")
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs.legend(by_label.values(), by_label.keys())
    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--encounter_rate", type=float, default=0.6)
    parser.add_argument("--recovery_rate", type=float, default=0.4)
    parser.add_argument("--cost_infection", type=float, default=1)
    parser.add_argument("--cost_lockdown", type=float, default=2)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-6)
    args = parser.parse_args()
    main(args)
    print(0)
