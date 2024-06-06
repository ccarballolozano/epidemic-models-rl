import argparse
from datetime import datetime
import os

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.sir_with_confinements.social_optimum import compute_social_optimum_policy


Ns = [30]
encounter_probs_N = np.linspace(0.1, 1, 10)
recovery_probs_N = np.linspace(0.1, 1, 10)
costs_infection = np.concatenate((1 / np.linspace(1, 10, 5), np.linspace(1, 10, 5)))
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
        "encounter_probs_N": encounter_probs_N,
        "recovery_probs_N": recovery_probs_N,
        "costs_infection": costs_infection,
        "costs_lockdown": costs_lockdown,
        "discount_factor": discount_factor,
        "theta": theta,
    }
    mlflow.log_params(params)
    n_iter = (
        len(Ns)
        * len(encounter_probs_N)
        * len(recovery_probs_N)
        * len(costs_infection)
        * len(costs_lockdown)
    )
    iter = 0
    for N in Ns:
        for encounter_prob_N in encounter_probs_N:
            encounter_prob = encounter_prob_N / N
            for recovery_prob_N in recovery_probs_N:
                recovery_prob = recovery_prob_N / N
                for cost_infection in costs_infection:
                    for cost_lockdown in costs_lockdown:
                        iter += 1
                        logger.info(
                            f"Running configuration {iter}/{n_iter}, N={N}, encounter_prob_N={encounter_prob_N}, recovery_prob_N={recovery_prob_N}, cost_infection={cost_infection}, cost_lockdown={cost_lockdown}"
                        )
                        policy, V = compute_social_optimum_policy(
                            N,
                            encounter_prob,
                            recovery_prob,
                            cost_infection,
                            cost_lockdown,
                            discount_factor,
                            theta,
                        )
                        os.makedirs(os.path.join(output_dir, "V"), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, "policy"), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, "plot"), exist_ok=True)

                        np.save(
                            os.path.join(
                                output_dir,
                                "V",
                                f"V_{N}_{encounter_prob_N}_{recovery_prob_N}_{cost_infection}_{cost_lockdown}.npy",
                            ),
                            V,
                        )
                        np.save(
                            os.path.join(
                                output_dir,
                                "policy",
                                f"policy_{N}_{encounter_prob_N}_{recovery_prob_N}_{cost_infection}_{cost_lockdown}.npy",
                            ),
                            policy,
                        )
                        f, ax = plt.subplots()
                        for m_s, m_i in policy:
                            if policy[m_s, m_i] == 0:
                                ax.plot(m_s, m_i, "x", color="red", label="confinement")
                            else:
                                ax.plot(
                                    m_s, m_i, "o", color="green", label="max exposure"
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
                                f"plt_{N}_{encounter_prob_N}_{recovery_prob_N}_{cost_infection}_{cost_lockdown}.png",
                            )
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
