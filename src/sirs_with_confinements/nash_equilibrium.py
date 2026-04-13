import argparse

from loguru import logger
import numpy as np

from .best_response import compute_best_response_policy


def compute_nash_equilibrium(
    N: int,
    encounter_rate: float,
    recovery_rate: float,
    resusceptible_rate: float,
    vaccination_rate: float,
    cost_infection: float,
    cost_lockdown: float,
    discount_factor: float,
    theta: float,
    max_iterations: int,
) -> dict:
    states = [
        (x, m_s, m_i)
        for x in ["S", "I", "R"]
        for m_s in range(N)
        for m_i in range(N)
        if m_s + m_i <= N - 1
    ]
    policy = {(m_s, m_i): np.random.randint(0, 2) for (_, m_s, m_i) in states}
    k = 0
    while True:
        k += 1
        policy_not_i = policy.copy()
        policy_br, _ = compute_best_response_policy(
            N,
            encounter_rate,
            recovery_rate,
            resusceptible_rate,
            vaccination_rate,
            cost_infection,
            cost_lockdown,
            discount_factor,
            theta,
            policy_not_i,
        )
        # parse policy_br to policy
        policy = {k[1:]: v for k, v in policy_br.items() if k[0] == "S"}
        delta = np.max(
            np.abs(list({s: policy[s] - policy_not_i[s] for s in policy}.values()))
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
    return policy


def main(args):
    policy_nash = compute_nash_equilibrium(
        args.N,
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

    import matplotlib.pyplot as plt
    import os

    n_confinement_states = 0
    n_total_states = 0
    for m_s, m_i in policy_nash:
        if policy_nash[m_s, m_i] == 0:
            n_confinement_states += 1
        n_total_states += 1
    print(f"Total states: {n_total_states}")
    print(f"Number confinement states: {n_confinement_states}")
    print(f"Proportion of confinement states: {n_confinement_states/n_total_states}")
    f, ax = plt.subplots()
    for m_s, m_i in policy_nash:
        if policy_nash[m_s, m_i] == 0:
            ax.plot(m_s, m_i, "x", color="red", label="$\pi^{sne}=0$")
        else:
            ax.plot(m_s, m_i, "o", color="green", label="$\pi^{sne}=1$")
    # ax.set_title("Nash Equilibrium Policy. State $(M_S, M_I)$")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_by_label = {k: by_label[k] for k in sorted(by_label)}
    ax.legend(sorted_by_label.values(), sorted_by_label.keys(), fontsize=14)
    ax.set_xlabel("$M_S$", fontsize=14)
    ax.set_ylabel("$M_I$", fontsize=14)
    f.tight_layout()
    plt.savefig(
        os.path.join(
            f"sirs_nash_plt_{args.N}_{args.encounter_rate}_{args.recovery_rate}_{args.resusceptible_rate}_{args.vaccination_rate}_{args.cost_infection}_{args.cost_lockdown}.png",
        ),
        bbox_inches="tight",
    )
    plt.show(block=True)
    print(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--encounter_rate", type=float, default=0.6)
    parser.add_argument("--recovery_rate", type=float, default=0.4)
    parser.add_argument("--resusceptible_rate", type=float, default=0.2)
    parser.add_argument("--vaccination_rate", type=float, default=0.2)
    parser.add_argument("--cost_infection", type=float, default=1.0)
    parser.add_argument("--cost_lockdown", type=float, default=1.0)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--theta", type=float, default=1e-6)
    parser.add_argument("--max_iterations", type=int, default=1e6)
    args = parser.parse_args()
    main(args)
