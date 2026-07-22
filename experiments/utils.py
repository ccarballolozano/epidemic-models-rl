"""Utilities for fetching and plotting experiment results from MLflow / AzureML."""

from __future__ import annotations

import logging
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ARTIFACTS_ROOT = Path("artifacts")

Q_LEARNING_LABEL = "Q-Learning"
QL_ABS_LABEL = "QL-ABS"

LEARN_MODE_STYLES: list[tuple[str, str, str]] = [
    ("complete", Q_LEARNING_LABEL, "tab:blue"),
    ("two_stages", QL_ABS_LABEL, "tab:orange"),
]


def _format_ka_label(label: str) -> str:
    """Render K^a using matplotlib mathtext in legend labels."""
    if "$K^{a}$" in label:
        return label
    return label.replace("K^a", "$K^{a}$")


# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------

_VALID_LEARN_MODES = {"complete", "two_stages"}


def validate_runs(
    runs: pd.DataFrame,
    *,
    expected_n: int,
    reinfection: bool,
    label: str = "",
) -> None:
    """Warn if any run in *runs* has unexpected parameter values.

    Checks
    ------
    - ``params.size`` == *expected_n* for every run
    - ``params.resusceptible_rate`` == ``"0.0"`` iff *reinfection* is False
    - ``tags.learn_mode`` is in ``{"complete", "two_stages"}`` for every run

    A single :func:`warnings.warn` is issued per violated check, listing the
    offending run IDs so they can be investigated without stopping execution.

    Parameters
    ----------
    runs:
        The filtered runs DataFrame returned by ``mlflow.search_runs``.
    expected_n:
        Expected population size (e.g. 5, 15, 50).
    reinfection:
        ``True`` if SIRS (resusceptible_rate > 0), ``False`` if SIR (= 0).
    label:
        Short human-readable name used in warning messages (e.g. ``"N=5 SIRS"``).
    """
    prefix = f"[{label}] " if label else ""

    # ── N (params.size) ───────────────────────────────────────────────────
    if "params.size" in runs.columns:
        wrong = runs[runs["params.size"].astype(str) != str(expected_n)]
        if not wrong.empty:
            warnings.warn(
                f"{prefix}{len(wrong)} run(s) have params.size != {expected_n}: "
                + ", ".join(wrong["run_id"].tolist()),
                stacklevel=2,
            )
    else:
        warnings.warn(f"{prefix}Column 'params.size' not found in runs DataFrame.", stacklevel=2)

    # ── Reinfection (params.resusceptible_rate) ───────────────────────────
    if "params.resusceptible_rate" in runs.columns:
        rate_str = runs["params.resusceptible_rate"].astype(str)
        is_zero = rate_str == "0.0"
        if reinfection:
            # Expect rate > 0 — flag any that are zero
            wrong = runs[is_zero]
            if not wrong.empty:
                warnings.warn(
                    f"{prefix}{len(wrong)} run(s) expected SIRS (resusceptible_rate > 0) "
                    f"but have resusceptible_rate = 0.0: "
                    + ", ".join(wrong["run_id"].tolist()),
                    stacklevel=2,
                )
        else:
            # Expect rate == 0 — flag any that are non-zero
            wrong = runs[~is_zero]
            if not wrong.empty:
                warnings.warn(
                    f"{prefix}{len(wrong)} run(s) expected SIR (resusceptible_rate = 0.0) "
                    f"but have non-zero resusceptible_rate: "
                    + ", ".join(wrong["run_id"].tolist()),
                    stacklevel=2,
                )
    else:
        warnings.warn(
            f"{prefix}Column 'params.resusceptible_rate' not found in runs DataFrame.",
            stacklevel=2,
        )

    # ── learn_mode ────────────────────────────────────────────────────────
    if "tags.learn_mode" in runs.columns:
        wrong = runs[~runs["tags.learn_mode"].isin(_VALID_LEARN_MODES)]
        if not wrong.empty:
            bad_modes = wrong["tags.learn_mode"].unique().tolist()
            warnings.warn(
                f"{prefix}{len(wrong)} run(s) have unexpected tags.learn_mode "
                f"{bad_modes}: " + ", ".join(wrong["run_id"].tolist()),
                stacklevel=2,
            )
    else:
        warnings.warn(f"{prefix}Column 'tags.learn_mode' not found in runs DataFrame.", stacklevel=2)


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
    max_workers: int = 10,
) -> pd.DataFrame:
    """Fetch metric history for each run; return a tidy DataFrame with learn_mode.

    MLflow calls are dispatched in parallel (network I/O bound) and results are
    accumulated in a list before a single ``pd.concat``, avoiding the O(n²)
    cost of growing a DataFrame on every iteration.

    ``max_workers`` is capped at 10 to match urllib3's default connection pool
    size for the AzureML endpoint — more threads than pool slots triggers
    "Connection pool is full" warnings and redundant TCP handshakes.
    """
    run_list = list(runs.itertuples())
    parts: list[pd.DataFrame] = []
    lock = threading.Lock()

    def _fetch(run) -> None:
        m = fetch_metric_history(client, run.run_id, metric_name)
        m["run_id"] = run.run_id
        with lock:
            parts.append(m)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(run_list) or 1)) as ex:
        futures = [ex.submit(_fetch, run) for run in run_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching metrics"):
            future.result()

    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    result = result.merge(runs[["run_id", "tags.learn_mode"]], on="run_id", how="left")
    return result


def fetch_metrics_for_runs_value_function(
    client,
    runs: pd.DataFrame,
    metric_name: str = "log_mean_relative_error_V",
    max_workers: int = 10,
) -> pd.DataFrame:
    """Fetch value-function metric history for each run; return a tidy DataFrame with learn_mode.

    Parameters
    ----------
    metric_name : str
        Default is "log_mean_relative_error_V" (value function error).
        Alternative: "log_max_relative_error_V".
    max_workers : int
        Parallel MLflow fetch threads. Capped at 10 to match urllib3's default
        connection pool size for the AzureML endpoint.
    """
    run_list = list(runs.itertuples())
    parts: list[pd.DataFrame] = []
    lock = threading.Lock()

    def _fetch(run) -> None:
        m = fetch_metric_history(client, run.run_id, metric_name)
        m["run_id"] = run.run_id
        with lock:
            parts.append(m)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(run_list) or 1)) as ex:
        futures = [ex.submit(_fetch, run) for run in run_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching metrics"):
            future.result()

    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
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
    title: str | None = None,
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
        (complete_df, Q_LEARNING_LABEL, "tab:blue"),
        (two_stage_df, QL_ABS_LABEL, "tab:orange"),
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
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# Palette for multiple Smart Q-learning (two-stage) K^a series (colorblind-friendly)
_MULTI_KA_COLORS = [
    "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def plot_mean_mre_multi_ka(
    complete_df: pd.DataFrame,
    two_stage_series: list[tuple[str, pd.DataFrame]],
    title: str | None = None,
    confidence: float = 0.95,
    use_t: bool = False,
    ax=None,
    y_label: str = "Log Mean Relative Error",
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
        Optional plot title. If None/empty, no title is shown.
    confidence:
        CI confidence level (default 0.95).
    ax:
        Optional existing Axes; a new figure is created when None.
    y_label:
        Y-axis label. Default "Log Mean Relative Error" (Q-functions).
        Use "Log Mean Relative Error (V)" for value functions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if not complete_df.empty:
        steps, mean, lo, hi = compute_mean_and_ci(
            complete_df, confidence=confidence, use_t=use_t
        )
        ax.plot(steps, mean, label=Q_LEARNING_LABEL, color="tab:blue")
        ax.fill_between(steps, lo, hi, alpha=0.2, color="tab:blue")

    for (label, df), color in zip(two_stage_series, cycle(_MULTI_KA_COLORS)):
        if df.empty:
            continue
        steps, mean, lo, hi = compute_mean_and_ci(
            df, confidence=confidence, use_t=use_t
        )
        ax.plot(
            steps,
            mean,
            label=f"{QL_ABS_LABEL} ({_format_ka_label(label)})",
            color=color,
        )
        ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    if title:
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


def load_run_value_functions(
    run_id: str,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
) -> dict | None:
    """Load Q_true and all Q checkpoints; return raw value functions.

    Returns ``{"V_true": ndarray(S+1,S+1), "checkpoints": {step: ndarray(S+1,S+1)}}``
    or ``None`` when no artifacts are found.  ``V = max(Q, axis=-1)``.
    """
    try:
        artifacts_path = get_local_artifacts_path(run_id, artifacts_root)
    except FileNotFoundError:
        return None

    q_true_files = list((artifacts_path / "Q_true").glob("*.npy"))
    if not q_true_files:
        return None

    Q_true = np.load(q_true_files[0])  # (S+1, S+1, A)
    V_true = np.max(Q_true, axis=-1)  # (S+1, S+1)

    checkpoints = {}
    for chkpt_dir in sorted(artifacts_path.glob("chkpt_*")):
        # The npy filename mirrors the dir name: chkpt_A_B_C -> Q_A_B_C.npy
        # Constructing the path directly avoids 1000 inner glob calls per run.
        suffix = chkpt_dir.name[len("chkpt"):]  # e.g. "_0_1000_1000"
        npy_path = chkpt_dir / f"Q{suffix}.npy"
        if not npy_path.exists():
            continue
        step = int(re.search(r"_(\d+)$", chkpt_dir.name).group(1))
        Q_ckpt = np.load(npy_path)  # (S+1, S+1, A)
        checkpoints[step] = np.max(Q_ckpt, axis=-1)  # (S+1, S+1)

    return {"V_true": V_true, "checkpoints": dict(sorted(checkpoints.items()))}


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
        suffix = chkpt_dir.name[len("chkpt"):]
        npy_path = chkpt_dir / f"Q{suffix}.npy"
        if not npy_path.exists():
            continue
        step = int(re.search(r"_(\d+)$", chkpt_dir.name).group(1))
        Q_ckpt = np.load(npy_path)  # (S+1, S+1, A)
        rel_err = np.abs(safe_Q_true - Q_ckpt) / np.abs(safe_Q_true)
        mre = np.nanmean(rel_err, axis=2)  # (S+1, S+1)
        results[step] = np.where(mre > 0, np.log10(mre), np.nan)

    return dict(sorted(results.items()))


def load_run_state_value_errors(
    run_id: str,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
) -> dict | None:
    """Load Q_true and all Q checkpoints; return ``{step: log10_mre_matrix}`` for value functions.

    Computes value function (V = max_a Q) relative error per state, not per state-action pair.
    The matrix has shape ``(S+1, S+1)`` where ``S`` is the population size.
    Each entry [m_s, m_i] is log10(|V_true[m_s,m_i] - V_ckpt[m_s,m_i]| / |V_true[m_s,m_i]|).
    """
    try:
        artifacts_path = get_local_artifacts_path(run_id, artifacts_root)
    except FileNotFoundError:
        return None

    q_true_files = list((artifacts_path / "Q_true").glob("*.npy"))
    if not q_true_files:
        return None

    Q_true = np.load(q_true_files[0])  # (S+1, S+1, A)
    V_true = np.max(Q_true, axis=-1)  # (S+1, S+1)
    safe_V_true = np.where(V_true != 0, V_true, np.nan)

    results = {}
    for chkpt_dir in sorted(artifacts_path.glob("chkpt_*")):
        suffix = chkpt_dir.name[len("chkpt"):]
        npy_path = chkpt_dir / f"Q{suffix}.npy"
        if not npy_path.exists():
            continue
        step = int(re.search(r"_(\d+)$", chkpt_dir.name).group(1))
        Q_ckpt = np.load(npy_path)  # (S+1, S+1, A)
        V_ckpt = np.max(Q_ckpt, axis=-1)  # (S+1, S+1)
        rel_err = np.abs(safe_V_true - V_ckpt) / np.abs(safe_V_true)
        results[step] = np.where(rel_err > 0, np.log10(rel_err), np.nan)

    return dict(sorted(results.items()))


# ---------------------------------------------------------------------------
# Parallel batch loading helpers
# ---------------------------------------------------------------------------


def load_runs_state_value_errors(
    runs: pd.DataFrame,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
    max_workers: int = 8,
) -> dict:
    """Load state value errors for all runs in *runs* in parallel.

    Returns a dict ``{run_id: {"learn_mode": str, "errors": {step: np.ndarray}}}``.

    Parallelism is effective here because ``np.load`` releases the GIL during
    the underlying file I/O, so multiple threads can load different run
    artifacts concurrently.
    """
    run_rows = [(row["run_id"], row["tags.learn_mode"]) for _, row in runs.iterrows()]
    results: dict = {}
    lock = threading.Lock()

    def _load(run_id: str, learn_mode: str) -> None:
        errors = load_run_state_value_errors(run_id, artifacts_root)
        if errors is not None:
            with lock:
                results[run_id] = {"learn_mode": learn_mode, "errors": errors}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(run_rows) or 1)) as ex:
        futures = [ex.submit(_load, rid, lm) for rid, lm in run_rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading state errors"):
            future.result()

    return results


def load_runs_value_functions(
    runs: pd.DataFrame,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
    max_workers: int = 8,
) -> dict:
    """Load value functions for all runs in *runs* in parallel.

    Returns a dict ``{run_id: {"learn_mode": str, "V_true": np.ndarray, "checkpoints": dict}}``.
    """
    run_rows = [(row["run_id"], row["tags.learn_mode"]) for _, row in runs.iterrows()]
    results: dict = {}
    lock = threading.Lock()

    def _load(run_id: str, learn_mode: str) -> None:
        vf = load_run_value_functions(run_id, artifacts_root)
        if vf is not None:
            with lock:
                results[run_id] = {"learn_mode": learn_mode, **vf}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(run_rows) or 1)) as ex:
        futures = [ex.submit(_load, rid, lm) for rid, lm in run_rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading value functions"):
            future.result()

    return results


def compute_per_state_series(
    per_state_errors: dict,
    learn_mode: str,
    m_s: int,
    m_i: int,
    confidence: float = 0.95,
    use_t: bool = False,
) -> tuple:
    """Compute mean (across runs) and CI of log10 RE for state ``(m_s, m_i)``."""
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
    states: tuple[int, int] | list[tuple[int, int]],
    size_label: str = "",
    confidence: float = 0.95,
    use_t: bool = False,
    show_title: bool = True,
    y_label: str = "Log Relative Error",
    suptitle_suffix: str = "Per-State $\\log_{10}$ RE",
) -> tuple:
    """Plot per-state log10 RE for Q-learning vs multiple Smart Q-learning K^a variants.

    Parameters
    ----------
    complete_per_state_errors:
        Per-state errors dict (keyed by run_id) for the ``complete`` runs.
    two_stage_series:
        List of ``(label, per_state_errors_dict)`` tuples, one per K^a value.
    states:
        A single ``(m_s, m_i)`` tuple or a list of tuples.  Each state gets
        its own subplot when multiple states are provided.
    size_label:
        Optional string appended to the plot title(s).
    show_title:
        When False, suppress subplot titles and figure suptitle.
    y_label:
        Y-axis label. Default "Log Relative Error" (Q-functions).
        Use "Log Relative Error (V)" for value functions.
    suptitle_suffix:
        Figure suptitle. Default "Per-State $\\log_{10}$ RE".
    """
    # Normalise to a list of (m_s, m_i) tuples
    if isinstance(states, tuple) and len(states) == 2 and isinstance(states[0], int):
        states = [states]

    n = len(states)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False
    )

    for idx, (m_s, m_i) in enumerate(states):
        ax = axes[idx // ncols][idx % ncols]

        steps, means, lo, hi = compute_per_state_series(
            complete_per_state_errors,
            "complete",
            m_s,
            m_i,
            confidence=confidence,
            use_t=use_t,
        )
        if steps is not None:
            ax.plot(steps, means, label=Q_LEARNING_LABEL, color="tab:blue")
            ax.fill_between(steps, lo, hi, alpha=0.2, color="tab:blue")

        for (label, per_state_errors), color in zip(two_stage_series, cycle(_MULTI_KA_COLORS)):
            steps, means, lo, hi = compute_per_state_series(
                per_state_errors,
                "two_stages",
                m_s,
                m_i,
                confidence=confidence,
                use_t=use_t,
            )
            if steps is None:
                continue
            ax.plot(
                steps,
                means,
                label=f"{QL_ABS_LABEL} ({_format_ka_label(label)})",
                color=color,
            )
            ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

        ax.set_xlabel("Step")
        ax.set_ylabel(y_label)
        if show_title:
            title = f"State $({m_s},\\,{m_i})$"
            if size_label:
                title += f",  {size_label}"
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if show_title:
        fig.suptitle(suptitle_suffix, y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_per_state_error(
    per_state_errors: dict,
    m_s: int,
    m_i: int,
    size_label: str = "",
    confidence: float = 0.95,
    use_t: bool = False,
    learn_modes: list | None = None,
    show_title: bool = True,
    ax=None,
    y_label: str = "Log Relative Error",
) -> tuple:
    """Plot log10 RE evolution with CI for state ``(m_s, m_i)`` across learn modes.

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
    y_label:
        Y-axis label. Default "Log Relative Error" (Q-functions).
        Use "Log Relative Error (V)" for value functions.
    """
    if learn_modes is None:
        learn_modes = LEARN_MODE_STYLES
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    for learn_mode, label, color in learn_modes:
        steps, means, lo, hi = compute_per_state_series(
            per_state_errors, learn_mode, m_s, m_i, confidence=confidence, use_t=use_t
        )
        if steps is None:
            continue
        ax.plot(steps, means, label=label, color=color)
        ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    if show_title:
        title = f"Per-State $\\log_{{10}}$ RE — State $({m_s},\\,{m_i})$"
        if size_label:
            title += f",  {size_label}"
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if created_fig:
        plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Raw value-function helpers
# ---------------------------------------------------------------------------


def compute_per_state_value_series(
    per_run_vf: dict,
    learn_mode: str,
    m_s: int,
    m_i: int,
    confidence: float = 0.95,
    use_t: bool = False,
) -> tuple:
    """Compute mean and CI of V(m_s, m_i) across runs over steps.

    Parameters
    ----------
    per_run_vf:
        Dict ``{run_id: {"learn_mode": str, "V_true": ndarray, "checkpoints": {step: ndarray}}}``.

    Returns
    -------
    (steps, means, lo, hi) — all None when no matching runs are found.
    """
    run_data = {
        rid: v for rid, v in per_run_vf.items() if v["learn_mode"] == learn_mode
    }
    if not run_data:
        return None, None, None, None

    all_steps = sorted(
        set(s for v in run_data.values() for s in v["checkpoints"].keys())
    )
    values_by_step: dict[int, list] = {}
    for step in all_steps:
        vals = [
            float(v["checkpoints"][step][m_s, m_i])
            for v in run_data.values()
            if step in v["checkpoints"]
        ]
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


def plot_per_state_value_multi_ka(
    complete_vf: dict,
    two_stage_vf_series: list[tuple[str, dict]],
    states: tuple[int, int] | list[tuple[int, int]],
    size_label: str = "",
    confidence: float = 0.95,
    use_t: bool = False,
    log_values: bool = False,
    show_title: bool = True,
) -> tuple:
    """Plot V(m_s, m_i) evolution for Q-learning and Smart Q-learning K^a variants.

    A horizontal dashed black line marks V_true(m_s, m_i) (the optimum).
    Mirrors :func:`plot_per_state_error_multi_ka`.

    Parameters
    ----------
    complete_vf:
        Per-run value-function dict for ``complete`` (Q-learning) runs.
    two_stage_vf_series:
        List of ``(label, per_run_vf)`` tuples, one per K^a value.
    states:
        Single ``(m_s, m_i)`` tuple or a list of tuples.
    size_label:
        Optional string appended to subplot titles.
    log_values:
        When True, plot ``log10(|V|)`` instead of raw V.  Useful when values
        span several orders of magnitude.
    """
    if isinstance(states, tuple) and len(states) == 2 and isinstance(states[0], int):
        states = [states]

    def _transform(arr):
        if log_values:
            return np.log10(np.abs(arr))
        return arr

    n = len(states)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False
    )

    # Gather V_true values per state from available runs
    all_vf = dict(complete_vf)
    for _, vf in two_stage_vf_series:
        all_vf.update(vf)

    for idx, (m_s, m_i) in enumerate(states):
        ax = axes[idx // ncols][idx % ncols]

        # Horizontal reference: mean V_true across runs
        true_vals = [
            float(v["V_true"][m_s, m_i]) for v in all_vf.values() if "V_true" in v
        ]
        if true_vals:
            ref = _transform(np.array([np.mean(true_vals)]))[0]
            ax.axhline(
                ref,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Optimum",
            )

        steps, means, lo, hi = compute_per_state_value_series(
            complete_vf, "complete", m_s, m_i, confidence=confidence, use_t=use_t
        )
        if steps is not None:
            ax.plot(steps, _transform(means), label=Q_LEARNING_LABEL, color="tab:blue")
            ax.fill_between(
                steps, _transform(lo), _transform(hi), alpha=0.2, color="tab:blue"
            )

        for (label, vf), color in zip(two_stage_vf_series, cycle(_MULTI_KA_COLORS)):
            steps, means, lo, hi = compute_per_state_value_series(
                vf, "two_stages", m_s, m_i, confidence=confidence, use_t=use_t
            )
            if steps is None:
                continue
            ax.plot(
                steps,
                _transform(means),
                label=f"{QL_ABS_LABEL} ({_format_ka_label(label)})",
                color=color,
            )
            ax.fill_between(
                steps, _transform(lo), _transform(hi), alpha=0.2, color=color
            )

        ax.set_xlabel("Step")
        ylabel = (
            f"$\\log_{{10}}|V({m_s},{m_i})|$" if log_values else f"$V({m_s},{m_i})$"
        )
        ax.set_ylabel(ylabel)
        if show_title:
            title = f"State $({m_s},\\,{m_i})$"
            if size_label:
                title += f",  {size_label}"
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if show_title:
        fig.suptitle("Per-State Value Function", y=1.01)
    plt.tight_layout()
    return fig, axes
