import argparse
from datetime import datetime
import os

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.sirs_with_confinements.social_optimum import compute_social_optimum_policy


NS = [30]
ENCOUNTER_RATES = [0.01, 0.1, 1, 10]
RECOVERY_RATES = [0.01, 0.1, 1, 10]
RESUSCEPTIBLE_RATES = [0, 0.01, 0.1, 1.0, 10]
VACCINATION_RATES = [0, 0.01, 0.1, 1.0, 10]
COSTS_INFECTION = [0.01, 0.1, 1.0, 5.0, 10, 100]
COSTS_LOCKDOWN = [1.0]

DISCOUNT_FACTOR = 0.99
THETA = 1e-6


def main(args):
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(output_dir, "social_optimum", str(datetime.now()))

    # force param types for output names formatting purposes
    Ns = [int(el) for el in NS]
    encounter_rates = [float(el) for el in ENCOUNTER_RATES]
    recovery_rates = [float(el) for el in RECOVERY_RATES]
    resusceptible_rates = [float(el) for el in RESUSCEPTIBLE_RATES]
    vaccination_rates = [float(el) for el in VACCINATION_RATES]
    costs_infection = [float(el) for el in COSTS_INFECTION]
    costs_lockdown = [float(el) for el in COSTS_LOCKDOWN]

    os.makedirs(output_dir, exist_ok=True)
    params = {
        "Ns": Ns,
        "encounter_rates": encounter_rates,
        "recovery_rates": recovery_rates,
        "resusceptible_rates": resusceptible_rates,
        "vaccination_rates": vaccination_rates,
        "costs_infection": costs_infection,
        "costs_lockdown": costs_lockdown,
        "discount_factor": DISCOUNT_FACTOR,
        "theta": THETA,
    }
    mlflow.log_params(params)
    n_iter = (
        len(Ns)
        * len(encounter_rates)
        * len(recovery_rates)
        * len(resusceptible_rates)
        * len(vaccination_rates)
        * len(costs_infection)
        * len(costs_lockdown)
    )
    os.makedirs(os.path.join(output_dir, "V"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "policy"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plot"), exist_ok=True)
    iter = 0
    for N in Ns:
        for encounter_rate in encounter_rates:
            for recovery_rate in recovery_rates:
                for resusceptible_rate in resusceptible_rates:
                    for vaccination_rate in vaccination_rates:
                        for cost_infection in costs_infection:
                            for cost_lockdown in costs_lockdown:
                                iter += 1
                                logger.info(
                                    f"Running configuration {iter}/{n_iter}, N={N}, encounter_rate={encounter_rate}, recovery_rate={recovery_rate}, resusceptible_rate={resusceptible_rate}, vaccination_rate={vaccination_rate}, cost_infection={cost_infection}, cost_lockdown={cost_lockdown}"
                                )
                                policy, V = compute_social_optimum_policy(
                                    N,
                                    encounter_rate,
                                    recovery_rate,
                                    resusceptible_rate,
                                    vaccination_rate,
                                    cost_infection,
                                    cost_lockdown,
                                    DISCOUNT_FACTOR,
                                    THETA,
                                )
                                np.save(
                                    os.path.join(
                                        output_dir,
                                        "V",
                                        f"V_{N}_{encounter_rate}_{recovery_rate}_{resusceptible_rate}_{vaccination_rate}_{cost_infection}_{cost_lockdown}.npy",
                                    ),
                                    V,
                                )
                                np.save(
                                    os.path.join(
                                        output_dir,
                                        "policy",
                                        f"policy_{N}_{encounter_rate}_{recovery_rate}_{resusceptible_rate}_{vaccination_rate}_{cost_infection}_{cost_lockdown}.npy",
                                    ),
                                    policy,
                                )
                                f, ax = plt.subplots()
                                for m_s, m_i in policy:
                                    if policy[m_s, m_i] == 0:
                                        ax.plot(
                                            m_s,
                                            m_i,
                                            "x",
                                            color="red",
                                            label="confinement",
                                        )
                                    else:
                                        ax.plot(
                                            m_s,
                                            m_i,
                                            "o",
                                            color="green",
                                            label="max exposure",
                                        )
                                ax.set_title("Social Optimum Policy")
                                ax.set_xlabel("Susceptible")
                                ax.set_ylabel("Infected")
                                handles, labels = ax.get_legend_handles_labels()
                                by_label = dict(zip(labels, handles))
                                ax.legend(by_label.values(), by_label.keys())
                                plt.savefig(
                                    os.path.join(
                                        output_dir,
                                        "plot",
                                        f"plt_{N}_{encounter_rate}_{recovery_rate}_{resusceptible_rate}_{vaccination_rate}_{cost_infection}_{cost_lockdown}.png",
                                    )
                                )
                                plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
