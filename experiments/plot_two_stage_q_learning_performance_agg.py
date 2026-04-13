import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Define folders for each algorithm type
two_stage_q_learning_folders = [
    "../outputs/aml_metrics/Two-stage Q-learning",
]

q_learning_folders = [
    "../outputs/aml_metrics/Q-learning",
    "../outputs/aml_metrics/funny_shirt_zcqxknsr2h",
    "../outputs/aml_metrics/olive_evening_t0wm540v06",
    "../outputs/aml_metrics/bold_pillow_6k94shb4vq",
]

# Add more folders by extending the lists above with actual folder paths
# Example:
# two_stage_q_learning_folders = [
#     "../outputs/aml_metrics/Two-stage Q-learning",
#     "../outputs/aml_metrics/two_stage_run_2",
#     "../outputs/aml_metrics/two_stage_run_3",
# ]
# q_learning_folders = [
#     "../outputs/aml_metrics/Q-learning",
#     "../outputs/aml_metrics/q_learning_run_2",
#     "../outputs/aml_metrics/q_learning_run_3",
# ]

algorithm_folders = {
    "Our approach ($K^{a}=35000$)": two_stage_q_learning_folders,
    "Q-learning": q_learning_folders,
}

metric_file_name = "log_relative_error_mean_chart_data.tsv"

# Create a DataFrame to hold all data
all_data = pd.DataFrame()

# Read data from all folders for each algorithm type
for alg_name, folders in algorithm_folders.items():
    for folder_idx, folder in enumerate(folders):
        file_path = os.path.join(folder, metric_file_name)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, sep="\t")
            # data = data[data["run_id"] == folder.split("/")[-1]] #Name was changed
            # data = data[data["Step"] > 0]
            data["alg_name"] = alg_name
            data["run_idx"] = folder_idx  # Track which run this is
            all_data = pd.concat([all_data, data])
        else:
            print(f"Warning: File {file_path} not found")

# Calculate mean and standard deviation for each algorithm across runs
aggregated_data = []
for alg_name in all_data["alg_name"].unique():
    alg_data = all_data[all_data["alg_name"] == alg_name]

    # Group by Step and calculate statistics across runs
    stats = (
        alg_data.groupby("Step")["log_relative_error_mean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats["alg_name"] = alg_name
    stats["std"] = stats["std"].fillna(0)  # Handle cases with only one run
    aggregated_data.append(stats)

# Combine all aggregated data
plot_data = pd.concat(aggregated_data, ignore_index=True)

# Apply Savitzky-Golay filter for smoothing (optional)
# for alg_name in plot_data["alg_name"].unique():
#     mask = plot_data["alg_name"] == alg_name
#     plot_data.loc[mask, "mean"] = savgol_filter(
#         plot_data.loc[mask, "mean"], window_length=11, polyorder=2
#     )

# Define line styles and colors
line_styles = ["-", "--"]
line_colors = ["tab:blue", "tab:pink"]

# Create the plot
f, ax = plt.subplots(figsize=(10, 6))

for i, alg_name in enumerate(plot_data["alg_name"].unique()):
    data_subset = plot_data[plot_data["alg_name"] == alg_name]

    # Plot the mean line
    ax.plot(
        data_subset["Step"],
        data_subset["mean"],
        label=alg_name,
        linestyle=line_styles[i % len(line_styles)],
        color=line_colors[i % len(line_colors)],
        linewidth=2,
    )

    # Add standard deviation bands
    ax.fill_between(
        data_subset["Step"],
        data_subset["mean"] - data_subset["std"],
        data_subset["mean"] + data_subset["std"],
        color=line_colors[i % len(line_colors)],
        alpha=0.2,
        label=f"{alg_name} ± std",
    )

# Add grid with soft lines
ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.5)

# x label must start at 0.0, no space in the left
ax.axis(xmin=0.0, xmax=1e6)
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
ax.xaxis.get_offset_text().set_fontsize(16)

# Create custom legend - only show algorithm names, not std bands
handles, labels = ax.get_legend_handles_labels()
# Keep only the main algorithm lines (not the std bands)
main_handles = [handles[i] for i in range(0, len(handles), 2)]
main_labels = [labels[i] for i in range(0, len(labels), 2)]
ax.legend(main_handles, main_labels, title="", fontsize=18)

# plt.title("Log Mean Relative Error over Steps")
ax.set_xlabel("Step ($k$)", fontsize=18)
ax.set_ylabel("Log Mean Relative Error", fontsize=18)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
f.tight_layout()

plt.savefig(
    os.path.join(f"compare_two_stage_q_learning_performance_agg_plt.pdf"),
    format="pdf",
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(f"compare_two_stage_q_learning_performance_agg_plt.eps"),
    format="eps",
    bbox_inches="tight",
)
plt.show(block=True)
