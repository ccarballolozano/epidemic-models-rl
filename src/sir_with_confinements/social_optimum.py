import argparse
from collections import defaultdict
import os
from datetime import datetime

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def compute_social_optimum_policy(
    N: int,
    encounter_prob: float,
    recovery_prob: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
    max_iterations: int = 1e6,
) -> list[dict, dict]:

    q_I = lambda m_s, m_i, action: encounter_prob * action * m_s * m_i / N
    q_R = lambda m_s, m_i, action: recovery_prob * m_i
    q_hat = lambda m_s, m_i, action: 1 - q_I(m_s, m_i, action) - q_R(m_s, m_i, action)

    states = [
        (m_s, m_i) for m_s in range(N + 1) for m_i in range(N + 1) if m_s + m_i <= N
    ]
    V = defaultdict(lambda: 0, {state: 0 for state in states})

    cost = lambda m_s, m_i, action: (cost_lockdown - action) * (
        m_s / N
    ) + cost_infection * (m_i / N)
    next_expected_value = (
        lambda m_s, m_i, action, V: q_I(m_s, m_i, action) * V[m_s - 1, m_i + 1]
        + q_R(m_s, m_i, action) * V[m_s, m_i - 1]
        + q_hat(m_s, m_i, action) * V[m_s, m_i]
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
            logger.info(f"Converged after {k} iterations, Delta: {delta}")
            break
        elif k >= max_iterations:
            logger.warning(
                f"Max iterations reached: {max_iterations} without convergence, Delta: {delta}"
            )
            break
        else:
            if k % 100 == 0:
                logger.debug(f"Iteration: {k}, Delta: {delta}")

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
    policy, V = compute_social_optimum_policy(
        args.N,
        args.encounter_prob,
        args.recovery_prob,
        args.cost_infection,
        args.cost_lockdown,
        args.discount_factor,
        args.theta,
    )

    logger.info(f"Policy: {policy}")
    logger.info(f"Value function: {V}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--encounter_prob", type=float, default=0.6)
    parser.add_argument("--recovery_prob", type=float, default=0.4)
    parser.add_argument("--cost_infection", type=float, default=6.7)
    parser.add_argument("--cost_lockdown", type=float, default=2)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-6)
    parser.add_argument("--max_iterations", type=int, default=1e6)
    args = parser.parse_args()
    main(args)
    print(0)
