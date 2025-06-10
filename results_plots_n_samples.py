import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from matplotlib.ticker import LogFormatter
import matplotlib.patches as patches



palette = sns.color_palette("colorblind", 8)  # Adjustable for various color themes

sns.set_context("paper", font_scale=1.5)  # Scale to increase font size

# If additional precision is needed, manually adjust elements
plt.rc('axes', titlesize=16)         # Larger title font size if using titles
plt.rc('axes', labelsize=16)         # Axis labels font size
plt.rc('xtick', labelsize=14)         # X-tick labels font size
plt.rc('ytick', labelsize=14)         # Y-tick labels font size
plt.rc('legend', fontsize=12)         # Legend font size
plt.rc('font', size=12)              # Base font size
plt.rc('axes', labelcolor='black')        # Makes x and y axis labels black
plt.rc('xtick', color='black')            # Makes x-tick labels black
plt.rc('ytick', color='black')            # Makes y-tick labels black
plt.rc('xtick', labelcolor='black')       # Makes x-tick numbers black
plt.rc('ytick', labelcolor='black')       # Makes y-tick numbers black
plt.rc('axes', edgecolor='black')         # Sets the axis edges to black
plt.rc('legend', labelcolor='black')  # Ensures legend text is black
plt.rcParams.update({
     'text.usetex': False,
#     'font.family': 'serif',
})

parent_dir = r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/"

# Load the data
stats_path = os.path.join(parent_dir, "results_n_samples/results_unlearning_best_per_model_by_aus.csv")
df = pd.read_csv(stats_path)

# Metrics as rows, methods as columns
metrics = [
    ("val_test_fgt_acc", "Val Test Forget Accuracy"),
    ("val_test_retain_acc", "Val Test Retain Accuracy"),
    ("AUS", "AUS")
]

n_rows = len(metrics)
n_cols = 3

# melt these
melted_df = df.melt(id_vars=["dataset", "method", "model", "model_num", "source", "samples_per_class", "Forget Class"],
                     value_vars=[metric[0] for metric in metrics],
                     var_name="metric",
                     value_name="value")

g = sns.relplot(
    data=melted_df,
    x="samples_per_class",
    y="value",
    hue="method",
    style="method",
    markers=True,
    dashes=True,
    row="metric",
    col="dataset",
    col_order=["cifar10", "cifar100", "tinyimagenet"],  
    facet_kws={'sharex': False, 'sharey': False},
    kind="line",
    err_style="band",
    errorbar=('ci', 95),
    markeredgewidth=0,
    linewidth=2,
)



def plot_with_black_box(plot_obj, filename):
    # Check if `plot_obj` is a Seaborn FacetGrid or a Matplotlib Figure
    if hasattr(plot_obj, 'axes') and isinstance(plot_obj, plt.Figure):
        # Handle Matplotlib Figure with individual axes
        for ax in plot_obj.axes:
            pos = ax.get_position()
            rect = patches.Rectangle(
                (pos.x0, pos.y0), pos.width, pos.height,
                linewidth=1.5, edgecolor='black', facecolor='none',
                transform=plot_obj.transFigure
            )
            plot_obj.add_artist(rect)
    elif hasattr(plot_obj, 'axes'):
        # Handle Seaborn FacetGrid with `.axes.flat`
        for ax in plot_obj.axes.flat:
            pos = ax.get_position()
            rect = patches.Rectangle(
                (pos.x0, pos.y0), pos.width, pos.height,
                linewidth=1.5, edgecolor='black', facecolor='none',
                transform=plt.gcf().transFigure
            )
            plt.gcf().add_artist(rect)
    
    # Save and show the plot
    plt.savefig(f"results_n_samples/{filename}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"results_n_samples/{filename}.pdf", dpi=600, bbox_inches='tight')

    plt.show()

for col, ax in enumerate(g.axes[0]):  # Loop through top row only
    title = ax.get_title().split("=")[-1].strip()  # Extract dataset name only
    # Custom formatting for specific datasets
    if title.lower() == "cifar10":
        display_title = "CIFAR10"
    elif title.lower() == "cifar100":
        display_title = "CIFAR100"
    elif title.lower() == "tinyimagenet":
        display_title = "TinyImageNet"
    ax.set_title(display_title)  # Bold the dataset name without "dataset="
    #ax.set_title(r"$\mathbf{" + display_title + "}$")  # Bold the dataset name without "dataset="


for i in range(1, g.axes.shape[0]):  # from row 1 to end
    for ax in g.axes[i]:
        ax.set_title("")    
        
# Manually set the same y-limits for all axes in the same row
for row_idx in range(g.axes.shape[0]):
    # Collect all y-limits for this row
    y_mins, y_maxs = [], []
    for ax in g.axes[row_idx]:
        ymin, ymax = ax.get_ylim()
        y_mins.append(ymin)
        y_maxs.append(ymax)
    
    # Compute global min and max for the row
    row_ymin = min(y_mins)
    row_ymax = max(y_maxs)
    
    # Apply the limits to all plots in this row
    for ax in g.axes[row_idx]:
        ax.set_ylim(row_ymin, row_ymax)


if g._legend is not None:
    g._legend.remove() 

for i, ax_row in enumerate(g.axes):
    if i == 0:  
        ax_row[0].set_ylabel("Forget Test Accuracy %")
        #ax_row[0].set_ylabel("$\mathcal{A}_f^t$ (Forget Test Accuracy \%)")
    elif i == 1:  
        ax_row[0].set_ylabel("Retain Test Accuracy %")
        #ax_row[0].set_ylabel("$\mathcal{A}_r^t$ (Retain Test Accuracy \%)")
    elif i == 2:  
        ax_row[0].set_ylabel("AUS")
    for ax in ax_row[1:]:
        ax.set_ylabel("")

    for ax in ax_row:
        # Set x-axis label only for bottom row
        if i == len(g.axes) - 1:
            ax.set_xlabel("Number of Samples per Class")
        else:
            ax.set_xlabel("")
        
handles, labels = g.axes[0, 0].get_legend_handles_labels()  # Get from the first subplot
g.axes[0, 2].legend(handles=handles, labels=labels, loc='upper right', frameon=True)

# Apply x-axis formatting by column index
for row in range(len(g.row_names)):
    for col, dataset in enumerate(g.col_names):
        ax = g.axes[row, col]
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(LogFormatter(base=10))
        ax.grid(True)

        if dataset == "cifar10":
            ax.set_xlim(1,6000)  
        elif dataset == "cifar100":
            ax.set_xlim(1,600)  
        elif dataset == "tinyimagenet":
            ax.set_xlim(1,600)  

# Save the plot before showing it
plot_with_black_box(g, "plot_n_sample")

