import argparse

from loguru import logger
import numpy as np

from .best_response import compute_best_response_policy


def compute_nash_equilibrium(
    N: int,
    encounter_prob: float,
    recovery_prob: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
    max_iterations: int,
) -> dict:
    states = [
        (x, m_s, m_i)
        for x in ["S", "I"]
        for m_s in range(N)
        for m_i in range(N)
        if m_s + m_i <= N - 1
    ]
    policy_not_i = {state: np.random.randint(0, 1) for state in states}
    k = 0
    while True:
        k += 1
        policy_bar = policy_not_i.copy()
        policy, _ = compute_best_response_policy(
            N,
            encounter_prob,
            recovery_prob,
            cost_infection,
            cost_lockdown,
            discount_factor,
            theta,
            policy_bar,
        )
        # parse policy_br to policy
        delta = np.max(
            np.abs(list({s: policy[s] - policy_bar[s] for s in policy}.values()))
        )
        if delta < theta:
            logger.info(
                f"Nash Equilibrium computation - Converged after {k} iterations, Delta: {delta}"
            )
            break
        elif k >= max_iterations:
            logger.warning(
                f"Nash Equilibrium computation - Max iterations reached: {max_iterations} without convergence, Delta: {delta}"
            )
            policy = None
            break
        else:
            if k % 1000 == 0:
                logger.debug(
                    f"Nash Equilibrium computation - Iteration: {k}, Delta: {delta}"
                )
        policy_not_i = policy.copy()
    return policy


def main(args):
    policy_nash = compute_nash_equilibrium(
        args.N,
        args.encounter_prob,
        args.recovery_prob,
        args.cost_infection,
        args.cost_lockdown,
        args.discount_factor,
        args.theta,
        args.max_iterations,
    )
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(1, 2)
    for x, m_s, m_i in policy_nash:
        if x == "S":
            if policy_nash[x, m_s, m_i] == 0:
                ax[0].plot(m_s, m_i, "x", color="red", label="confinement")
            else:
                ax[0].plot(m_s, m_i, "o", color="green", label="max exposure")
    ax[0].set_title(
        "Nash Equilibrium Policy. State $('S', M_S, M_I), M_S + M_I <= N-1$"
    )
    ax[0].set_xlabel("Susceptible")
    ax[0].set_ylabel("Infected")
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys())
    for x, m_s, m_i in policy_nash:
        if x == "S":
            if policy_nash[x, m_s, m_i] == 0:
                ax[1].plot(m_s + 1, m_i, "x", color="red", label="confinement")
            else:
                ax[1].plot(m_s + 1, m_i, "o", color="green", label="max exposure")
        [
            ax[1].plot(0, m_i, "x", color="red", label="confinement")
            for m_i in range(args.N + 1)
        ]
    ax[1].set_title("Nash Equilibrium Policy. State $(M_S, M_I), M_S + M_I <= N$")
    ax[1].set_xlabel("Susceptible")
    ax[1].set_ylabel("Infected")
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), by_label.keys())
    plt.show()
    print(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--encounter_prob", type=float, default=0.1)
    parser.add_argument("--recovery_prob", type=float, default=0.1)
    parser.add_argument("--cost_infection", type=float, default=1.0)
    parser.add_argument("--cost_lockdown", type=float, default=1.0)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--theta", type=float, default=1e-6)
    parser.add_argument("--max_iterations", type=int, default=1e6)
    args = parser.parse_args()
    main(args)
