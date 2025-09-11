import pandas as pd
from collections import defaultdict, Counter

# Load both ResNet-18 and ResNet-50 result files
resnet18_df = pd.read_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_fc_resnet18/results_mean_std_all_numeric_resnet18.csv")
resnet50_df = pd.read_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_fc_resnet50/results_mean_std_all_numeric_resnet50.csv")

# Add architecture column
resnet18_df["arch"] = "resnet18"
resnet50_df["arch"] = "resnet50"

# Combine the data
stats_df = pd.concat([resnet18_df, resnet50_df], ignore_index=True)

# Select key columns to display
columns_to_display = [
    ("val_test_retain_acc", "\mathcal{A}^t_r"),
    ("val_test_fgt_acc", "\mathcal{A}^t_f"),
    ("AUS", "AUS")
]

# === Helper to determine D_r-free and D_f-free flags
def get_data_free_flags(method, source):
    if method in ["original", "retrained"]:
        return ("--", "--")
    elif method in ["MM"]:
        return (r"\cmark", r"\cmark") 
    elif method in ["FT","RE"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\xmark", r"\cmark")
    elif method in ["NG", "RL", "BS", "BE", "LAU", "DELETE"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\cmark", r"\xmark")
    elif method in ["NGFTW", "DUCK", "SCRUB", "SCAR"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\xmark", r"\xmark")
    return (r"\xmark", r"\xmark")


# Group rows by dataset
datasets = stats_df["dataset"].unique()

# === Define display names and references
method_name_and_ref = {
    "original": ("Original", "–"),
    "retrained": (r"\makecell{Retrained (Full)}", "–"),
    "RE":        (r"\makecell{Retrained (FC)}", "–"),
    "FT": ("FT \citep{golatkar2020eternal}", "–"),
    "NG": ("NG \citep{golatkar2020eternal}", "–"),
    "NGFTW": ("NG+ \citep{kurmanji2023towards}", "–"),
    "RL": ("RL \citep{hayase2020selective}", "–"),
    "BS": ("BS \citep{chen2023boundary}", "–"),
    "BE": ("BE \citep{chen2023boundary}", "–"),
    "LAU": ("LAU \citep{kim2024layer}", "–"),
    "SCRUB": ("SCRUB \citep{kurmanji2023towards}", "–"),
    "DUCK": ("DUCK \citep{cotogni2023duck}", "–"),
    "SCAR": ("SCAR \citep{bonato2024retain}", "–"),
    "DELETE": ("DELETE \citep{zhou2025decoupled}", "–"),


}


method_order = ["original", "retrained", "RE", "FT", "NG", "RL","BS", "BE", "DELETE", "LAU", "NGFTW", "SCRUB", "DUCK", "SCAR"]



def sort_key(key):
    method_part = key.split(" (")[0]
    source_part = key.split(" (")[1].split(")")[0]
    arch_part = key.split("[")[-1].replace("]", "")
    method_idx = method_order.index(method_part) if method_part in method_order else len(method_order)
    source_idx = 0 if source_part == "real" else 1
    arch_idx = 0 if arch_part == "resnet18" else 1
    return (arch_idx, method_idx, source_idx)

grouped_methods = defaultdict(lambda: {"CIFAR10": ["-"]*3, "CIFAR100": ["-"]*3, "TinyImageNet": ["-"]*3})
access_flags = {}
max_min_tracker = defaultdict(lambda: defaultdict(dict))
method_counts = Counter()

# Track best values for highlighting
for arch in ["resnet18", "resnet50"]:
    for dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        df_filtered = stats_df[
            (stats_df["dataset"].str.lower().str.contains(dataset.lower())) &
            (stats_df["arch"] == arch) &
            (stats_df["method"] != "DUCK")
        ]
        for metric, label in columns_to_display:
            metric_mean = f"{metric}_mean"
            if df_filtered.empty:
                continue
            if "retain" in metric or metric == "AUS":
                max_min_tracker[arch][dataset][label] = df_filtered[metric_mean].max()
            elif "fgt" in metric:
                max_min_tracker[arch][dataset][label] = df_filtered[metric_mean].min()

# Process rows
for _, row in stats_df.iterrows():
    if row["method"] == "DUCK":
        continue
    method = row["method"]
    source = row["source"]
    dataset = row["dataset"].strip().lower()
    dataset = "CIFAR10" if dataset == "cifar10" else "CIFAR100" if dataset == "cifar100" else "TinyImageNet" if "tiny" in dataset else None
    if dataset is None:
        continue

    key = f"{method} ({source}) [{row['arch']}]"
    method_counts[(method, row['arch'])] += 1
    values = []
    for prefix, label in columns_to_display:
        mean_col = f"{prefix}_mean"
        std_col = f"{prefix}_std"
        val, std = row.get(mean_col), row.get(std_col, 0.0)
        if pd.isna(val):
            values.append("-")
            continue
        val_str = f"{val:.3f}" if label == "AUS" else f"{val:.2f}" if label == "\mathcal{A}^t_r" else f"{val:.1f}"
        std_str = f"{std:.3f}" if label == "AUS" else f"{std:.2f}" if label == "\mathcal{A}^t_r" else f"{std:.1f}"
        if round(val, 3) == round(max_min_tracker[row["arch"]][dataset][label], 3) and label in [r"\mathcal{A}^t_r", "AUS"]:
            val_str = f"\\textbf{{{val_str}}}"
        values.append(f"{val_str}\\scriptsize{{\\,$\\pm$\\,{std_str}}}")


    grouped_methods[key][dataset] = values
    grouped_methods[key]["arch"] = row["arch"]
    access_flags[key] = get_data_free_flags(method, source)

# Generate LaTeX for each dataset
for dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
    latex_table = r"""\begin{table*}[h]
\centering
\captionsetup{font=small}
\caption{Class unlearning performance comparison on """ + dataset + r""" using ResNet-18 and ResNet-50 as the base architecture.
Rows highlighted in gray represent our results using synthetic data, while the corresponding non-shaded rows use original samples with the same method.
Columns $\mathcal{D}_r$-free and $\mathcal{D}_f$-free indicate whether the method operates without access to the retain or forget set, respectively, with (\cmark) denoting true and (\xmark) denoting false.}
\label{tab:main_results_""" + dataset.lower() + r"""}
\resizebox{\textwidth}{!}{
\begin{tabular}{c|cc|ccc|ccc}
\toprule
\toprule
 \multirow{2}{*}{Method} & \multirow{2}{*}{\shortstack{$\mathcal{D}_r$ \\ free}} & \multirow{2}{*}{\shortstack{$\mathcal{D}_f$ \\ free}} & \multicolumn{3}{c|}{\textbf{ResNet-18}} & \multicolumn{3}{c}{\textbf{ResNet-50}} \\
 &  &  & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS $\uparrow$ & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS $\uparrow$ \\
\midrule
\midrule
"""

    prev_base_method = None

    # Gather all (method, source) pairs that exist in the data
    # Use a dummy arch to apply sort_key (we’ll use resnet18 for sorting priority)
    base_keys = sorted(set(
        (k.split(" (")[0], k.split(" (")[1].split(")")[0])
        for k in grouped_methods.keys()
    ), key=lambda k: sort_key(f"{k[0]} ({k[1]}) [resnet18]"))

    
    for base_method, source in base_keys:

        key18 = f"{base_method} ({source}) [resnet18]"
        key50 = f"{base_method} ({source}) [resnet50]"
    
        method_display, _ = method_name_and_ref.get(base_method, (base_method, "–"))

        if base_method in ["original", "retrained"] and source == "real":
            D_r_free, D_f_free = access_flags.get(key18, access_flags.get(key50, ("--", "--")))
            values18 = grouped_methods.get(key18, {}).get(dataset, ["-"] * 3)
            values50 = grouped_methods.get(key50, {}).get(dataset, ["-"] * 3)
    
            # Create multirow cells for the method and access flags
            method_cell = rf"\multirow{{2}}{{*}}{{{method_display}}}"
            dr_free = rf"\multirow{{2}}{{*}}{{{D_r_free}}}"
            df_free = rf"\multirow{{2}}{{*}}{{{D_f_free}}}"
    
            # Add multirow values for each column from ResNet-18 and ResNet-50
            val_cells = [rf"\multirow{{2}}{{*}}{{{v}}}" for v in values18 + values50]
    
            row1 = [method_cell, dr_free, df_free] + val_cells
            row2 = ["", "", ""] + [""] * len(val_cells)
    
            latex_table += " & ".join(row1) + r" \\" + "\n"
            latex_table += " & ".join(row2) + r" \\" + "\n"
    
            if base_method == "original":
                latex_table += r"\midrule" + "\n" + r"\midrule" + "\n"    
    
            prev_base_method = base_method
            continue


            
                    
        if (base_method != prev_base_method):
            if prev_base_method in ["FT", "DELETE"]:
                latex_table += r"\midrule" + "\n" + r"\midrule" + "\n"
            else:
                latex_table += r"\midrule" + "\n"
        prev_base_method = base_method        
        

        
        
        # === Only display method name once across real/synth rows ===
        num_rows = sum(
            1 for src in ["real", "synth"]
            if f"{base_method} ({src}) [resnet18]" in grouped_methods or f"{base_method} ({src}) [resnet50]" in grouped_methods
        )
        method_cell = (
            rf"\multirow{{{num_rows}}}{{*}}{{{method_display}}}"
            if source == "real" else ""
        )
    
        D_r_free, D_f_free = access_flags.get(key18, access_flags.get(key50, ("--", "--")))
    
        values_resnet18 = grouped_methods[key18][dataset] if key18 in grouped_methods else ["-"] * len(columns_to_display)
        values_resnet50 = grouped_methods[key50][dataset] if key50 in grouped_methods else ["-"] * len(columns_to_display)
    
        row = [method_cell, D_r_free, D_f_free] + values_resnet18 + values_resnet50
    
        if source == "synth":
            row = [row[0]] + [rf"\cellcolor{{gray!15}}{cell}" for cell in row[1:]]
    
        latex_table += " & ".join(row) + r" \\" + "\n"


    latex_table += r"""\bottomrule
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    with open(f"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_fc/table_total_random_fc_{dataset}.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"✅ LaTeX table saved to table_total_random_fc_{dataset}.tex")
