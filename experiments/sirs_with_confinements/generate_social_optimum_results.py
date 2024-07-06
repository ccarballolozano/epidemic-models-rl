import argparse
from datetime import datetime
import os

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.sirs_with_confinements.social_optimum import compute_social_optimum_policy


Ns = [30]
encounter_rates = [0.01, 0.1, 1, 10, 100]
recovery_rates = [0.01, 0.1, 1, 10, 100]
susceptible_rates = [0, 0.01, 0.1, 1, 10, 100]
vaccination_rates = [0, 0.01, 0.1, 1, 10, 100]
costs_infection = [0.01, 0.1, 1, 10, 100]
costs_lockdown = [1]

discount_factor = 0.99
theta = 1e-6


def main(args):
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(output_dir, "social_optimum", str(datetime.now()))
    os.makedirs(output_dir, exist_ok=True)
    params = {
        "Ns": Ns,
        "encounter_rates": encounter_rates,
        "recovery_rates": recovery_rates,
        "susceptible_rates": susceptible_rates,
        "vaccination_rates": vaccination_rates,
        "costs_infection": costs_infection,
        "costs_lockdown": costs_lockdown,
        "discount_factor": discount_factor,
        "theta": theta,
    }
    mlflow.log_params(params)
    n_iter = (
        len(Ns)
        * len(encounter_rates)
        * len(recovery_rates)
        * len(susceptible_rates)
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
                for susceptible_rate in susceptible_rates:
                    for vaccination_rate in vaccination_rates:
                        for cost_infection in costs_infection:
                            for cost_lockdown in costs_lockdown:
                                iter += 1
                                logger.info(
                                    f"Running configuration {iter}/{n_iter}, N={N}, encounter_rate={encounter_rate}, recovery_rate={recovery_rate}, susceptible_rate={susceptible_rate}, vaccination_rate={vaccination_rate}, cost_infection={cost_infection}, cost_lockdown={cost_lockdown}"
                                )
                                policy, V = compute_social_optimum_policy(
                                    N,
                                    encounter_rate,
                                    recovery_rate,
                                    susceptible_rate,
                                    vaccination_rate,
                                    cost_infection,
                                    cost_lockdown,
                                    discount_factor,
                                    theta,
                                )
                                np.save(
                                    os.path.join(
                                        output_dir,
                                        "V",
                                        f"V_{N}_{encounter_rate}_{recovery_rate}_{susceptible_rate}_{vaccination_rate}_{cost_infection}_{cost_lockdown}.npy",
                                    ),
                                    V,
                                )
                                np.save(
                                    os.path.join(
                                        output_dir,
                                        "policy",
                                        f"policy_{N}_{encounter_rate}_{recovery_rate}_{susceptible_rate}_{vaccination_rate}_{cost_infection}_{cost_lockdown}.npy",
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
                                        f"plt_{N}_{encounter_rate}_{recovery_rate}_{susceptible_rate}_{vaccination_rate}_{cost_infection}_{cost_lockdown}.png",
                                    )
                                )
                                plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
