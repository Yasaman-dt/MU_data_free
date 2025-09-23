import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib
from collections import OrderedDict

matplotlib.rcParams.update({
    'text.usetex': False
})
sns.set_theme(style="whitegrid") 
sns.set_context("notebook", font_scale=1) 
#palette = sns.color_palette("colorblind", 8)  # Adjustable for various color themes

sns.set_context("paper", font_scale=1.5)  # Scale to increase font size

# matplotlib.rcParams.update({
#     #'text.usetex': True,                # Use LaTeX for all text rendering
#     #'font.family': 'serif',            # Use serif fonts
#     #'font.serif': ['Computer Modern Roman'],  # Matches LaTeX default
#     'axes.labelsize': 18,
#     'font.size': 15,
#     'legend.fontsize': 15,
#     'xtick.labelsize': 17,
#     'ytick.labelsize': 17,
#     'axes.titlesize': 18
# })

# If additional precision is needed, manually adjust elements
plt.rc('axes', titlesize=17)         # Larger title font size if using titles
plt.rc('axes', labelsize=17)         # Axis labels font size
plt.rc('xtick', labelsize=14)         # X-tick labels font size
plt.rc('ytick', labelsize=14)         # Y-tick labels font size
plt.rc('legend', fontsize=13)         # Legend font size
plt.rc('font', size=12)              # Base font size
plt.rc('axes', labelcolor='black')        # Makes x and y axis labels black
plt.rc('xtick', color='black')            # Makes x-tick labels black
plt.rc('ytick', color='black')            # Makes y-tick labels black
plt.rc('xtick', labelcolor='black')       # Makes x-tick numbers black
plt.rc('ytick', labelcolor='black')       # Makes y-tick numbers black
plt.rc('axes', edgecolor='black')         # Sets the axis edges to black
plt.rc('legend', labelcolor='black')  # Ensures legend text is black

# plt.rcParams.update({
#      'text.usetex': False,
# #     'font.family': 'serif',
# })

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

method_order = ["DELETE", "NG", "NG+", "RL", "FT", "SCRUB"]
melted_df["method"] = pd.Categorical(melted_df["method"], categories=method_order, ordered=True)


g = sns.relplot(
    data=melted_df,
    x="samples_per_class",
    y="value",
    hue="method",
    style="method",
    hue_order=method_order,
    style_order=method_order,
    markers="o",
    dashes=True,
    palette='tab10',
    row="metric",
    col="dataset",
    col_order=["cifar10", "cifar100", "tinyimagenet"],  
    facet_kws={'sharex': False, 'sharey': False},
    kind="line",
    err_style="band",
    errorbar=('ci', 95),
    markeredgewidth=0,
    linewidth=3,
)
for ax in g.axes.flat:
    for collection in ax.collections:
        collection.set_alpha(0.1)
for ax in g.axes.flat:
    for line in ax.lines:
        line.set_markersize(8) 

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
                linewidth=1, edgecolor='black', facecolor='none',
                transform=plt.gcf().transFigure
            )
            plt.gcf().add_artist(rect)
    
    # Save and show the plot
    plt.savefig(f"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_n_samples/{filename}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_n_samples/{filename}.pdf", dpi=600, bbox_inches='tight')

    plt.show()

for col, ax in enumerate(g.axes[0]):  # Loop through top row only
    title = ax.get_title().split("=")[-1].strip()  # Extract dataset name only
    # Custom formatting for specific datasets
    if title.lower() == "cifar10":
        display_title = "CIFAR-10"
    elif title.lower() == "cifar100":
        display_title = "CIFAR-100"
    elif title.lower() == "tinyimagenet":
        display_title = "TinyImageNet"
    ax.set_title(r"$\mathbf{" + display_title + "}$")  # Bold the dataset name without "dataset="
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
        ax_row[0].set_ylabel("Forget Test Accuracy (%)")
        #ax_row[0].set_ylabel("$\mathcal{A}_f^t$ (Forget Test Accuracy \%)")
    elif i == 1:  
        ax_row[0].set_ylabel("Retain Test Accuracy (%)")
        #ax_row[0].set_ylabel("$\mathcal{A}_r^t$ (Retain Test Accuracy \%)")
    elif i == 2:  
        ax_row[0].set_ylabel("AUS")
    for ax in ax_row[1:]:
        ax.set_ylabel("")

    for ax in ax_row:
        # Set x-axis label only for bottom row
        if i == len(g.axes) - 1:
            ax.set_xlabel("# Samples per Class")
        else:
            ax.set_xlabel("")
        
handles, labels = g.axes[0, 0].get_legend_handles_labels()  # Get from the first subplot
g.axes[0, 2].legend(handles=handles, labels=labels, loc='upper right', frameon=True)



# Apply x-axis formatting by column index
for row in range(len(g.row_names)):
    for col, dataset in enumerate(g.col_names):
        ax = g.axes[row, col]
        ax.set_xscale("log")
        ax.minorticks_on()
        ax.tick_params(which='major', bottom=True, left =True)
        ax.tick_params(which='minor', bottom=True)
        ax.grid(True)

        if dataset == "cifar10":
            ax.set_xlim(1,6000)  
        elif dataset == "cifar100":
            ax.set_xlim(1,600)  
        elif dataset == "tinyimagenet":
            ax.set_xlim(1,600)  

# Save the plot before showing it
plot_with_black_box(g, "plot_n_sample_resnet18")
#plt.savefig(f"results_n_samples/plot_n_sample.png", dpi=600, bbox_inches='tight')



# --- Per-dataset row-of-3 figures (legend only inside AUS panel) ------------

metric_keys   = ["val_test_fgt_acc", "val_test_retain_acc", "AUS"]
metric_titles = {
    "val_test_fgt_acc":   "Forget Test Accuracy (%)",
    "val_test_retain_acc":"Retain Test Accuracy (%)",
    "AUS":                "AUS",
}
dataset_titles = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tinyimagenet": "TinyImageNet"}
xlims = {"cifar10": (1, 6000), "cifar100": (1, 600), "tinyimagenet": (1, 600)}

melted_df["metric"] = pd.Categorical(melted_df["metric"], categories=metric_keys, ordered=True)

# Stable style/marker mapping across all datasets
method_levels = list(pd.unique(melted_df["method"]))

for ds in ["cifar10", "cifar100", "tinyimagenet"]:
    sub = melted_df[melted_df["dataset"] == ds].copy()
    if sub.empty:
        continue

    
    g_ds = sns.relplot(
        data=sub,
        x="samples_per_class",
        y="value",
        hue="method",             
        style="method",
        hue_order=method_order,
        style_order=method_order,
        markers="o",
        dashes=True,
        palette="tab10",
        col="metric",              # 1 row × 3 columns
        col_order=metric_keys,
        facet_kws={"sharex": False, "sharey": False},
        kind="line",
        err_style="band",
        errorbar=("ci", 95),
        markeredgewidth=0,
        linewidth=3)


    for ax in g_ds.axes.flat:
        for collection in ax.collections:
            collection.set_alpha(0.1)
    for ax in g_ds.axes.flat:
        for line in ax.lines:
            line.set_markersize(8) 

    # no subplot titles
    for ax in g_ds.axes[0]:
        ax.set_title("")

    g_ds.fig.subplots_adjust(wspace=0.25)
                
    # axes styling
    for ax in g_ds.axes[0]:
        ax.set_xscale("log")
        ax.minorticks_on()
        ax.tick_params(which="major", bottom=True, left=True)
        ax.tick_params(which="minor", bottom=True, left=True)
        ax.set_xlabel("# Samples per Class")
        ax.grid(True)

    # dataset-specific x-range
    if ds in xlims:
        for ax in g_ds.axes[0]:
            ax.set_xlim(*xlims[ds])

    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    # force integer ticks on y-axis
    for ax in g_ds.axes[0]:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # integers only
        ax.yaxis.set_major_formatter(ScalarFormatter())         # remove .0
    
    # y labels (independent ranges per panel)
    g_ds.axes[0][0].set_ylabel("Forget Test Accuracy (%)")
    g_ds.axes[0][0].yaxis.set_label_position("left")
    g_ds.axes[0][0].yaxis.tick_left()
    
    g_ds.axes[0][1].set_ylabel("Retain Test Accuracy (%)")
    g_ds.axes[0][1].yaxis.set_label_position("left")
    g_ds.axes[0][1].yaxis.tick_left()
    
    g_ds.axes[0][2].set_ylabel("AUS")
    g_ds.axes[0][2].yaxis.set_label_position("left")
    g_ds.axes[0][2].yaxis.tick_left()

    fig_legend = g_ds._legend
    fig_legend.remove()
    
    # (optional) enforce your explicit order
    order = [labels.index(m) for m in method_order if m in labels]
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
      
    ax_af = g_ds.axes[0][0]
    ax_af.set_ylim(-3, 90)                                  # cap at 1.0
    ax_af.yaxis.set_major_locator(MultipleLocator(10))    # ticks every 5
    ax_af.yaxis.set_major_formatter(ScalarFormatter())   # no decimals
    
    # Retain accuracy (middle panel)
    ax_ar = g_ds.axes[0][1]
    ax_ar.set_ylim(-3, 90)                                  # cap at 1.0
    ax_ar.yaxis.set_major_locator(MultipleLocator(10))    
    ax_ar.yaxis.set_major_formatter(ScalarFormatter())   
    
    # AUS (right panel)
    ax_aus = g_ds.axes[0][2]
    ax_aus.set_ylim(0.5, 1.03)                                  # cap at 1.0
    ax_aus.yaxis.set_major_locator(MultipleLocator(0.1))       # ticks every 0.1
    ax_aus.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


    
    ax_last = g_ds.axes[0, 1]
    ax_last.legend(handles, labels, title="method",
                   loc="lower right", frameon=True, ncol=1)
    
    plot_with_black_box(g_ds, f"plot_n_sample_resnet18_{ds}")
