# Reinforcement Learning for Epidemic Control on the SIRS Model

A research codebase for learning optimal confinement policies in stochastic epidemic models.
The project casts epidemic control as a Markov Decision Process (MDP) defined on a
mean-field SIRS model and studies how well tabular Q-learning — with and without a
two-stage curriculum — can recover the social-optimum policy computed by exact dynamic
programming.

---

## Table of contents

1. [Background](#background)
2. [Repository structure](#repository-structure)
3. [Model](#model)
4. [Algorithms](#algorithms)
5. [Evaluation](#evaluation)
6. [Getting started](#getting-started)
7. [Running experiments](#running-experiments)
8. [Experiment tracking](#experiment-tracking)
9. [Notebook workflow](#notebook-workflow)
10. [Game-theoretic baselines](#game-theoretic-baselines)

---

## Background

The **SIRS model** is a continuous-time Markov chain over a finite population of *N*
individuals, each in one of three states: **S**usceptible, **I**nfected, or **R**ecovered.
Unlike the classic SIR model, recovered individuals can lose immunity and return to the
susceptible pool (rate δ), making recurrent epidemic waves possible.

This project studies the *mean-field* (aggregate) version of the model, where the state
is the pair (M_S, M_I) counting susceptible and infected individuals in the population.
A social planner — or a learned agent — observes this aggregate state and chooses at
each step whether to impose a confinement measure (action = 1) or not (action = 0).
Confinement scales down the encounter rate, reducing transmission at the cost of an
economic penalty proportional to the susceptible population.

Two families of policies are studied:

| Policy | Description |
|---|---|
| **Social optimum** | Minimises the discounted cumulative cost over the whole population — computed exactly via Q-value iteration. |
| **Q-learning** | Learns the same objective from simulated trajectories; several curricula are compared. |

The game-theoretic module (`src/sirs_with_confinements/`) also contains a **Nash
equilibrium** baseline where each player selfishly minimises their own cost given the
behaviour of others.

---

## Repository structure

```
epidemic-models-rl/
│
├── src/
│   ├── envs/
│   │   └── env.py                  # SIRSEnv — Gymnasium environment
│   ├── rl/
│   │   └── q_iteration.py          # Exact Q-value iteration (social optimum)
│   ├── sirs_with_confinements/     # Game-theoretic module (Nash / social optimum)
│   │   ├── best_response.py
│   │   ├── nash_equilibrium.py
│   │   └── social_optimum.py
│   ├── algorithms.py               # Tabular Q-learning with curriculum modes
│   ├── metrics.py                  # Value-function and policy error metrics
│   ├── misc.py                     # Q-table initialisation utilities
│   ├── plot.py                     # Shared plotting helpers
│   ├── rl_experiment.py            # Single-run experiment entry point
│   ├── azureml_utils.py            # Azure ML helpers
│   └── generate_compare_q_learning_performance_experiments.py
│
├── experiments/
│   ├── utils.py                    # MLflow fetch, CI, and plotting utilities
│   ├── 01_rl_experiment.ipynb      # Introductory RL notebook
│   ├── experiment.ipynb            # General experiment notebook
│   ├── Compare_Two_Stage_Q_Learning_Performance.ipynb
│   ├── plot_two_stage_q_learning.py
│   ├── plot_two_stage_q_learning_performance.py
│   ├── plot_two_stage_q_learning_performance_agg.py
│   └── sirs_with_confinements/     # Game-theory experiment scripts
│
├── jobs/
│   └── smart_q_learning_job.yml    # Azure ML job definition
│
├── pyproject.toml                  # Project metadata and dependencies (uv)
├── environment.yml                 # Conda environment
└── azureml_environment.yml         # Azure ML environment spec
```

---

## Model

### State space

The environment state is the pair **(M_S, M_I)** where M_S ∈ {0, …, N} is the number
of susceptible individuals and M_I ∈ {0, …, N} is the number of infected individuals,
subject to M_S + M_I ≤ N.  The remaining N − M_S − M_I individuals are recovered.

The total number of valid states is (N+1)(N+2)/2.

### Action space

Binary: **0** (no confinement) or **1** (confinement).

### Transition dynamics

At each step exactly one of the following events can occur.  Let

```
u = 1 / [N × (β + γ + δ + ν)]
```

where β is the encounter rate, γ the recovery rate, δ the resusceptibility rate, and ν
the vaccination rate.  Then the transition probabilities are:

| Event | Probability | Next state |
|---|---|---|
| New infection | `u × β × a × M_S × M_I / N` | (M_S − 1, M_I + 1) |
| Vaccination | `u × ν × M_S` | (M_S − 1, M_I) |
| Recovery | `u × γ × M_I` | (M_S, M_I − 1) |
| Resusceptibility | `u × δ × M_R` | (M_S + 1, M_I) |
| No change | 1 − (sum above) | (M_S, M_I) |

Note that the infection probability is proportional to `a`, so confinement (a = 0)
completely prevents new infections.

### Cost function

The instantaneous cost in state (M_S, M_I) under action *a* is:

```
c(M_S, M_I, a) = (c_lock − a) × (M_S / N) + c_inf × (M_I / N)
```

where c_lock is the lockdown cost and c_inf is the infection cost.
The reward returned by the environment is `−c(s, a)`.

### Terminal states

States with M_I = 0 are absorbing when δ = 0 (no resusceptibility).
When δ > 0 the epidemic can always restart, so the episode length is controlled
by `max_steps_episode`.

---

## Algorithms

### Q-value iteration (exact social optimum)

`src/rl/q_iteration.py` implements the Bellman optimality operator directly on the
full state-action space.  It iterates until the sup-norm of the update falls below a
threshold θ:

```
Q(s, a) ← r(s, a) + γ × Σ_{s'} P(s'|s, a) × max_{a'} Q(s', a')
```

The resulting policy and Q-table are used as **ground truth** for all evaluations.

### Tabular Q-learning (`src/algorithms.py`)

Standard one-step Q-learning with ε-greedy exploration and an exponential learning-rate
schedule:

```
α(t) = α_min + (α_max − α_min) × exp(−α_decay × episode)
```

Four **learning modes** control episode initialisation and termination:

| Mode | Stage initialisation | Episode end |
|---|---|---|
| `complete` | Uniform over all valid states | After `max_steps_episode` steps |
| `fixed_no_infection` | Absorbing states (M_I = 0) only | When M_I = 0 |
| `independent_no_infection` | Uniform; separate treatment per type | When M_I returns to 0 |
| `two_stages` | **Stage 1**: absorbing states → **Stage 2**: transient states | Stage 1: custom; Stage 2: on absorbing entry |

The **two-stage curriculum** (`two_stages`) is the main proposed method.  In stage 1 the
agent learns the Q-values of absorbing states first, providing stable bootstrap targets
before the harder transient region is explored in stage 2.  Optional `alpha_restart_on_stage_change`
resets the learning-rate schedule at the stage transition.

#### Key hyperparameters

| Parameter | Description |
|---|---|
| `n_steps` | Total training steps |
| `first_stage_steps` | Steps allocated to stage 1 |
| `alpha_max` / `alpha_min` | Learning rate bounds |
| `alpha_decay` | Exponential decay constant |
| `epsilon` | ε-greedy exploration probability |
| `discount_factor` | γ — discount factor |
| `stage2_absorbing_extra_steps` | Extra steps inside absorbing states before episode ends in stage 2 |

---

## Evaluation

All metrics are computed against the exact Q-table produced by Q-value iteration and
only over valid states (M_S + M_I ≤ N).

| Metric | Symbol | Description |
|---|---|---|
| Mean absolute error (V) | MAE_V | Mean of \|V*(s) − V(s)\| |
| Max absolute error (V) | MaxAE_V | Max of \|V*(s) − V(s)\| |
| Mean relative error (V) | MRE_V | Mean of \|V* − V\| / \|V*\| |
| Mean absolute error (Q) | MAE_Q | Mean of \|Q*(s,a) − Q(s,a)\| |
| Suboptimal action rate | — | Proportion of states where argmax Q ≠ argmax Q* |

All metrics are logged to MLflow at every `log_every_n_steps` steps, enabling
learning-curve analysis across runs.

---

## Getting started

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) **or** conda

### Install with uv

```bash
# Clone the repository
git clone https://github.com/ccarballolozano/epidemic-models-rl.git
cd epidemic-models-rl

# Create environment and install dependencies
uv sync
```

### Install with conda

```bash
conda env create -f environment.yml
conda activate env
```

### Azure ML workspace (optional)

If you intend to run jobs on Azure ML, create a `config.json` file at the repository root
(this file is gitignored):

```json
{
    "subscription_id": "<your-subscription-id>",
    "resource_group":  "<your-resource-group>",
    "workspace_name":  "<your-workspace-name>"
}
```

Set the `TENANT_ID` environment variable (e.g. in a `.env` file) before running any
Azure ML scripts.

---

## Running experiments

### Compute the social optimum

```bash
cd src
python rl/q_iteration.py \
    --size 10 \
    --encounter_rate 1.1 \
    --recovery_rate 0.6 \
    --resusceptible_rate 0.3 \
    --vaccination_rate 0.2 \
    --cost_infection 2 \
    --cost_lockdown 1.001 \
    --discount_factor 0.99 \
    --theta 1e-12 \
    --output_dir ../outputs
```

### Run a single Q-learning experiment (local MLflow)

```bash
cd src
python rl_experiment.py \
    --size 10 \
    --encounter_rate 1.1 \
    --recovery_rate 0.6 \
    --resusceptible_rate 0.3 \
    --vaccination_rate 0.2 \
    --cost_infection 2 \
    --cost_lockdown 1.001 \
    --discount_factor 0.99 \
    --n_steps 1000000 \
    --n_episodes 1000000 \
    --alpha_max 0.5 \
    --alpha_min 0.0001 \
    --alpha_decay 0.0001 \
    --epsilon 0.1 \
    --max_steps_episode 2000 \
    --learn_mode two_stages \
    --first_stage_steps 300000 \
    --state_action_values_initialization random \
    --log_every_n_steps 1000 \
    --save_every_n_steps 100000 \
    --alpha_restart_on_stage_change \
    --stage2_absorbing_extra_steps 1 \
    --tag_run_group my_experiment
```

### Batch experiments on Azure ML

```bash
cd src
python generate_compare_q_learning_performance_experiments.py
```

This script submits parallel `complete` vs `two_stages` runs to Azure ML using the
parameters defined in `BASE_PARAMS` and sweeps over the modes.

### Open the MLflow UI

```bash
mlflow ui
```

Then navigate to `http://localhost:5000`.

---

## Experiment tracking

Every run logs the following to **MLflow**:

- All environment and algorithm hyperparameters as *params*
- The full learning curve of all evaluation metrics at every `log_every_n_steps` steps
- Q-table checkpoints (`.npy`) at every `save_every_n_steps` steps
- The ground-truth Q-table (`Q_true.npy`)
- Value-function and policy plots for the ground-truth policy

`experiments/utils.py` provides helper functions to fetch runs from MLflow (or Azure ML),
compute confidence-interval bands across seeds, and generate publication-quality figures
comparing `complete` vs `two_stages` performance.

---

## Notebook workflow

Notebooks are committed **without outputs** to keep the repository clean.
Local copies with outputs are stored as `<name>_with_outputs.ipynb` files, which are
gitignored.

To auto-strip outputs from notebooks on every future commit, install
[nbstripout](https://github.com/kynan/nbstripout):

```bash
pip install nbstripout
nbstripout --install   # installs the git filter defined in .gitattributes
```

From then on, outputs are stripped automatically at commit time — your local notebook
(with results) is never touched.

---

## Game-theoretic baselines

`src/sirs_with_confinements/` implements the individual-level SIRS model and three
solution concepts:

| Module | Concept | Description |
|---|---|---|
| `social_optimum.py` | Social optimum | Policy minimising the total population cost |
| `best_response.py` | Best response | Optimal policy for one player given all others' fixed policy |
| `nash_equilibrium.py` | Nash equilibrium | Fixed point of the best-response map — each player is individually rational |

These baselines are used in the `experiments/sirs_with_confinements/` scripts to
generate proportion-of-confinement and value-function comparisons between cooperative
and non-cooperative equilibria.

