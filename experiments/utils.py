"""Utilities for fetching and plotting experiment results from MLflow / AzureML."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ARTIFACTS_ROOT = Path("artifacts")

LEARN_MODE_STYLES: list[tuple[str, str, str]] = [
    ("complete", "Q-learning", "tab:blue"),
    ("two_stages", "Smart Q-learning", "tab:orange"),
]


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def fetch_metric_history(
    client,
    run_id: str,
    key: str,
) -> pd.DataFrame:
    """Fetch the full step-by-step history of *key* for *run_id*."""
    try:
        history = client.get_metric_history(run_id, key)
        logging.info(
            "Fetching %d entries for run_id %s and key %s", len(history), run_id, key
        )
        history = [metric.to_dictionary() for metric in history]
    except Exception:
        logging.exception(
            "Error fetching metric history for run_id %s and key %s", run_id, key
        )
        history = []
    return pd.DataFrame(history)


def fetch_metrics_for_runs(
    client,
    runs: pd.DataFrame,
    metric_name: str = "log_mean_relative_error",
) -> pd.DataFrame:
    """Fetch metric history for each run; return a tidy DataFrame with learn_mode."""
    result = pd.DataFrame()
    for run in tqdm(runs.itertuples(), total=len(runs)):
        m = fetch_metric_history(client, run.run_id, metric_name)
        m["run_id"] = run.run_id
        result = pd.concat([result, m], ignore_index=True)
    result = result.merge(runs[["run_id", "tags.learn_mode"]], on="run_id", how="left")
    return result


# ---------------------------------------------------------------------------
# Confidence-interval helpers
# ---------------------------------------------------------------------------


def compute_mean_and_ci(
    df: pd.DataFrame,
    confidence: float = 0.95,
    use_t: bool = False,
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, lo, hi) arrays from a tidy metric DataFrame.

    Parameters
    ----------
    confidence:
        Confidence level (default 0.95).
    use_t:
        When True, use the Student's t critical value (exact for small n).
        When False (default), use the normal (z) critical value — appropriate
        when the number of runs is large.
    """
    grouped = df.groupby("step")["value"]
    mean = grouped.mean()
    sem = grouped.sem()
    n = grouped.count()
    if use_t:
        h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        h = sem * stats.norm.ppf((1 + confidence) / 2)
    return mean.index, mean.values, (mean - h).values, (mean + h).values


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_mean_mre(
    complete_df: pd.DataFrame,
    two_stage_df: pd.DataFrame,
    title: str,
    confidence: float = 0.95,
    use_t: bool = False,
    ax=None,
) -> tuple:
    """Plot mean log MRE with CI bands for both learn modes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    for df, label, color in [
        (complete_df, "Q-learning", "tab:blue"),
        (two_stage_df, "Smart Q-learning", "tab:orange"),
    ]:
        if df.empty:
            continue
        steps, mean, lo, hi = compute_mean_and_ci(
            df, confidence=confidence, use_t=use_t
        )
        ax.plot(steps, mean, label=label, color=color)
        ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel("Log Mean Relative Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# Palette for multiple Smart Q-learning (two-stage) K^a series (colorblind-friendly)
_MULTI_KA_COLORS = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def plot_mean_mre_multi_ka(
    complete_df: pd.DataFrame,
    two_stage_series: list[tuple[str, pd.DataFrame]],
    title: str,
    confidence: float = 0.95,
    use_t: bool = False,
    ax=None,
) -> tuple:
    """Plot Q-learning vs multiple Smart Q-learning variants with different K^a values.

    Parameters
    ----------
    complete_df:
        Metric DataFrame for the ``complete`` (standard Q-learning) runs.
    two_stage_series:
        List of ``(label, df)`` tuples, one per K^a value,
        e.g. ``[("K^a = 105 000", df_105k), ("K^a = 150 000", df_150k)]``.
    title:
        Plot title.
    confidence:
        CI confidence level (default 0.95).
    ax:
        Optional existing Axes; a new figure is created when None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if not complete_df.empty:
        steps, mean, lo, hi = compute_mean_and_ci(
            complete_df, confidence=confidence, use_t=use_t
        )
        ax.plot(steps, mean, label="Q-learning", color="tab:blue")
        ax.fill_between(steps, lo, hi, alpha=0.2, color="tab:blue")

    for (label, df), color in zip(two_stage_series, _MULTI_KA_COLORS):
        if df.empty:
            continue
        steps, mean, lo, hi = compute_mean_and_ci(
            df, confidence=confidence, use_t=use_t
        )
        ax.plot(steps, mean, label=f"Smart Q-learning ({label})", color=color)
        ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel("Log Mean Relative Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------


def get_local_artifacts_path(
    run_id: str,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
) -> Path:
    """Return the local path to a downloaded run's artifacts, or raise FileNotFoundError."""
    local_path = Path(artifacts_root) / run_id
    if not local_path.exists():
        raise FileNotFoundError(
            f"Artifacts folder for run_id '{run_id}' not found at {local_path}."
        )
    return local_path


def load_run_state_errors(
    run_id: str,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
) -> dict | None:
    """Load Q_true and all Q checkpoints; return ``{step: log10_mre_matrix}`` or None.

    The matrix has shape ``(S+1, S+1)`` where ``S`` is the population size.
    Only Q checkpoint files (``Q_*.npy``) are loaded; ``state_update_counts_*.npy``
    files in the same directories are ignored.
    """
    try:
        artifacts_path = get_local_artifacts_path(run_id, artifacts_root)
    except FileNotFoundError:
        return None

    q_true_files = list((artifacts_path / "Q_true").glob("*.npy"))
    if not q_true_files:
        return None

    Q_true = np.load(q_true_files[0])  # (S+1, S+1, A)
    safe_Q_true = np.where(Q_true != 0, Q_true, np.nan)

    results = {}
    for chkpt_dir in sorted(artifacts_path.glob("chkpt_*")):
        npy_files = [f for f in chkpt_dir.glob("Q_*.npy")]
        if not npy_files:
            continue
        match = re.search(r"_(\d+)\.npy$", npy_files[0].name)
        if not match:
            continue
        step = int(match.group(1))
        Q_ckpt = np.load(npy_files[0])  # (S+1, S+1, A)
        rel_err = np.abs(safe_Q_true - Q_ckpt) / np.abs(safe_Q_true)
        mre = np.nanmean(rel_err, axis=2)  # (S+1, S+1)
        results[step] = np.where(mre > 0, np.log10(mre), np.nan)

    return dict(sorted(results.items()))


# ---------------------------------------------------------------------------
# Per-state error helpers
# ---------------------------------------------------------------------------


def compute_per_state_series(
    per_state_errors: dict,
    learn_mode: str,
    m_s: int,
    m_i: int,
    confidence: float = 0.95,
    use_t: bool = False,
) -> tuple:
    """Compute mean and CI of log10 MRE across runs for state ``(m_s, m_i)``."""
    run_errors = {
        rid: v["errors"]
        for rid, v in per_state_errors.items()
        if v["learn_mode"] == learn_mode
    }
    if not run_errors:
        return None, None, None, None

    all_steps = sorted(set(s for e in run_errors.values() for s in e.keys()))
    values_by_step: dict[int, list] = {}
    for step in all_steps:
        vals = [e[step][m_s, m_i] for e in run_errors.values() if step in e]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            values_by_step[step] = vals

    steps = sorted(values_by_step.keys())
    means = np.array([np.mean(values_by_step[s]) for s in steps])
    lo, hi = means.copy(), means.copy()
    for i, s in enumerate(steps):
        vals = values_by_step[s]
        n = len(vals)
        if n > 1:
            if use_t:
                h = stats.sem(vals) * stats.t.ppf((1 + confidence) / 2, n - 1)
            else:
                h = stats.sem(vals) * stats.norm.ppf((1 + confidence) / 2)
            lo[i] = means[i] - h
            hi[i] = means[i] + h
    return steps, means, lo, hi


def plot_per_state_error_multi_ka(
    complete_per_state_errors: dict,
    two_stage_series: list[tuple[str, dict]],
    m_s: int,
    m_i: int,
    size_label: str = "",
    confidence: float = 0.95,
    use_t: bool = False,
    ax=None,
) -> tuple:
    """Plot per-state log10 MRE for Q-learning vs multiple Smart Q-learning K^a variants.

    Parameters
    ----------
    complete_per_state_errors:
        Per-state errors dict (keyed by run_id) for the ``complete`` runs.
    two_stage_series:
        List of ``(label, per_state_errors_dict)`` tuples, one per K^a value.
    m_s, m_i:
        State to plot.
    size_label:
        Optional string appended to the plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    steps, means, lo, hi = compute_per_state_series(
        complete_per_state_errors, "complete", m_s, m_i,
        confidence=confidence, use_t=use_t,
    )
    if steps is not None:
        ax.plot(steps, means, label="Q-learning", color="tab:blue")
        ax.fill_between(steps, lo, hi, alpha=0.2, color="tab:blue")

    for (label, per_state_errors), color in zip(two_stage_series, _MULTI_KA_COLORS):
        steps, means, lo, hi = compute_per_state_series(
            per_state_errors, "two_stages", m_s, m_i,
            confidence=confidence, use_t=use_t,
        )
        if steps is None:
            continue
        ax.plot(steps, means, label=f"Smart Q-learning ({label})", color=color)
        ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

    title = f"Per-State $\\log_{{10}}$ MRE — State $({m_s},\\,{m_i})$"
    if size_label:
        title += f",  {size_label}"
    ax.set_xlabel("Step")
    ax.set_ylabel(f"$\\log_{{10}}$ MRE at $({m_s},{m_i})$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_per_state_error(
    per_state_errors: dict,
    m_s: int,
    m_i: int,
    size_label: str = "",
    confidence: float = 0.95,
    use_t: bool = False,
    learn_modes: list | None = None,
    ax=None,
) -> tuple:
    """Plot log10 MRE evolution with CI for state ``(m_s, m_i)`` across learn modes.

    Parameters
    ----------
    per_state_errors:
        Dict keyed by run_id, each value ``{"learn_mode": str, "errors": dict}``.
    m_s, m_i:
        State indices; must satisfy ``m_s + m_i <= size``.
    size_label:
        Optional string appended to the plot title, e.g. ``"Size = 5"``.
    confidence:
        Confidence level for the CI band (default 0.95).
    learn_modes:
        List of ``(learn_mode_str, label, color)`` tuples.
        Defaults to :data:`LEARN_MODE_STYLES`.
    ax:
        Existing Axes to draw on; a new figure is created when *None*.
    """
    if learn_modes is None:
        learn_modes = LEARN_MODE_STYLES
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    for learn_mode, label, color in learn_modes:
        steps, means, lo, hi = compute_per_state_series(
            per_state_errors, learn_mode, m_s, m_i, confidence=confidence, use_t=use_t
        )
        if steps is None:
            continue
        ax.plot(steps, means, label=label, color=color)
        ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

    title = f"Per-State $\\log_{{10}}$ MRE — State $({m_s},\\,{m_i})$"
    if size_label:
        title += f",  {size_label}"
    ax.set_xlabel("Step")
    ax.set_ylabel(f"$\\log_{{10}}$ MRE at $({m_s},{m_i})$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax
