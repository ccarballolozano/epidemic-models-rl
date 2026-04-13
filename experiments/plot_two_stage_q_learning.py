import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

experiments_folders = [
    "../outputs/aml_metrics/cyan_ship_sbszzycyph",
    "../outputs/aml_metrics/funny_pear_sczdyc4n5h",
    "../outputs/aml_metrics/sincere_ball_cs6t5wxqpc",
    "../outputs/aml_metrics/lime_orange_sfwyqgyxtm",
]
experiments_K_abs = [25000, 50000, 100000, 500000]

## N =30
# experiments_folders = [
#    "../outputs/aml_metrics/clever_boot_3zggsq03rx",
#    "../outputs/aml_metrics/stoic_brick_dnktfr9yx6",
#    "../outputs/aml_metrics/placid_avocado_lsg881yv8m",
#    "../outputs/aml_metrics/sincere_knot_x8p0l0d28d",
# ]
# experiments_K_abs = [0, 50000, 100000, 150000]

metric_file_name = "log_relative_error_mean_chart_data.tsv"

# Create a DataFrame to hold all data
all_data = pd.DataFrame()

for i, folder in enumerate(experiments_folders):
    file_path = os.path.join(folder, metric_file_name)
    data = pd.read_csv(file_path, sep="\t")
    data = data[data["run_id"] == folder.split("/")[-1]]
    data = data[data["Step"] > 0]
    data["K_abs"] = experiments_K_abs[i]
    all_data = pd.concat([all_data, data])

# Apply Savitzky-Golay filter for smoothing
all_data["log_relative_error_mean"] = all_data.groupby("K_abs")[
    "log_relative_error_mean"
].transform(lambda x: savgol_filter(x, window_length=11, polyorder=2))

# Create a new column for formatted K_abs
all_data["K_abs_label"] = all_data["K_abs"].apply(lambda x: f"$K^{{a}}={x}$")

# Define line styles
line_styles = [":", "--", "-", "-."]
##line_styles = None
# Plot using seaborn
f, ax = plt.subplots(figsize=(10, 6))
for i, (label, style) in enumerate(zip(all_data["K_abs_label"].unique(), line_styles)):
    sns.lineplot(
        data=all_data[all_data["K_abs_label"] == label],
        x="Step",
        y="log_relative_error_mean",
        label=label,
        linestyle=style,
        linewidth=2,  # Increase line width
        ax=ax,
    )

# Add grid with soft lines
ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.5)

# x label must start at 0.0, no space in the left
ax.axis(xmin=0.0, xmax=1e6)
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
ax.xaxis.get_offset_text().set_fontsize(16)
ax.legend(title="", fontsize=18)
# plt.title("Log Mean Relative Error over Steps")
ax.set_xlabel("Step ($k$)", fontsize=18)
ax.set_ylabel("Log Mean Relative Error", fontsize=18)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
f.tight_layout()
plt.savefig(
    os.path.join(
        f"compare_two_stage_q_learning_plt.png",
    ),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(f"compare_two_stage_q_learning_plt.eps"),
    format="eps",
    bbox_inches="tight",
)
plt.show(block=True)
