import argparse
import glob

import numpy as np
import pandas as pd


def main(args):
    social_optimum_dir = args.social_optimum_dir
    nash_equilibrium_dir = args.nash_equilibrium_dir
    social_optimum_policy_names = list(glob.glob(f"{social_optimum_dir}/policy_*.npy"))
    nash_equilibrium_policy_names = list(
        glob.glob(f"{nash_equilibrium_dir}/policy_*.npy")
    )

    params = [
        policy_name.split("/")[-1].split(".npy")[0].split("_")[1:]
        for policy_name in nash_equilibrium_policy_names
    ]
    N_ = sorted(list(set([str_to_num(param[0]) for param in params])))
    encounter_rate_ = sorted(list(set([str_to_num(param[1]) for param in params])))
    recovery_rate_ = sorted(list(set([str_to_num(param[2]) for param in params])))
    susceptible_rate_ = sorted(list(set([str_to_num(param[3]) for param in params])))  #
    vaccination_rate_ = sorted(list(set([str_to_num(param[4]) for param in params])))  #
    cost_infection_ = sorted(list(set([str_to_num(param[5]) for param in params])))
    cost_lockdown_ = sorted(list(set([str_to_num(param[6]) for param in params])))

    proportions_df = pd.DataFrame(
        columns=[
            "N",
            "encounter_rate",
            "recovery_rate",
            "susceptible_rate",
            "vaccination_rate",
            "cost_infection",
            "cost_lockdown",
            "social_optimum_confinement_proportion",
            "nash_equilibrium_confinement_proportion",
        ]
    )
    for N in N_:
        for encounter_rate in encounter_rate_:
            for recovery_rate in recovery_rate_:
                for susceptible_rate in susceptible_rate_:
                    for vaccination_rate in vaccination_rate_:
                        for cost_infection in cost_infection_:
                            for cost_lockdown in cost_lockdown_:
                                social_optimum_policy = load_policy(
                                    social_optimum_dir,
                                    N,
                                    encounter_rate,
                                    recovery_rate,
                                    susceptible_rate,
                                    vaccination_rate,
                                    cost_infection,
                                    cost_lockdown,
                                )
                                social_optimum_confinement_proportion = (
                                    compute_confinement_proportion(
                                        social_optimum_policy
                                    )
                                )
                                nash_equilibrium_policy = load_policy(
                                    nash_equilibrium_dir,
                                    N,
                                    encounter_rate,
                                    recovery_rate,
                                    susceptible_rate,
                                    vaccination_rate,
                                    cost_infection,
                                    cost_lockdown,
                                )
                                nash_equilibrium_policy = {
                                    (m_s, m_i): v
                                    for (
                                        x,
                                        m_s,
                                        m_i,
                                    ), v in nash_equilibrium_policy.items()
                                    if x == "S"
                                }
                                nash_equilibrium_confinement_proportion = (
                                    compute_confinement_proportion(
                                        nash_equilibrium_policy
                                    )
                                )
                                proportions_df.loc[len(proportions_df)] = {
                                    "N": N,
                                    "encounter_rate": encounter_rate,
                                    "recovery_rate": recovery_rate,
                                    "susceptible_rate": susceptible_rate,
                                    "vaccination_rate": vaccination_rate,
                                    "cost_infection": cost_infection,
                                    "cost_lockdown": cost_lockdown,
                                    "social_optimum_confinement_proportion": social_optimum_confinement_proportion,
                                    "nash_equilibrium_confinement_proportion": nash_equilibrium_confinement_proportion,
                                }
    proportions_df.reset_index(drop=True, inplace=True)
    proportions_df[proportions_df["nash_equilibrium_confinement_proportion"] > proportions_df["social_optimum_confinement_proportion"]]
    proportions_df[proportions_df["nash_equilibrium_confinement_proportion"] < proportions_df["social_optimum_confinement_proportion"]]
    proportions_df[proportions_df["nash_equilibrium_confinement_proportion"] == proportions_df["social_optimum_confinement_proportion"]]

    proportions_df.to_csv("outputs/confinement_proportions.csv", index=False)


def load_policy(
    policies_dir,
    N,
    encounter_rate,
    recovery_rate,
    susceptible_rate,
    vaccination_rate,
    cost_infection,
    cost_lockdown,
):
    try:
        file_path = f"{policies_dir}/policy_{N}_{float_to_int(encounter_rate)}_{float_to_int(recovery_rate)}_{float_to_int(susceptible_rate)}_{float_to_int(vaccination_rate)}_{float_to_int(cost_infection)}_{float_to_int(cost_lockdown)}.npy"
        policy = np.load(file_path, allow_pickle=True).item()
        return policy
    except Exception as e:
        raise ("Error loading policy:", e)


def compute_confinement_proportion(policy: dict):
    total = len(policy)
    confinement = sum([1 for v in policy.values() if v == 0])
    return confinement / total


def float_to_int(num):
    if int(num) == num:
        return int(num)
    else:
        return num


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nash_equilibrium_dir",
        type=str,
        default="outputs/nash_equilibrium/20240626/policy",
    )
    parser.add_argument(
        "--social_optimum_dir",
        type=str,
        default="outputs/social_optimum/20240626/policy",
    )
    args = parser.parse_args()
    main(args)
