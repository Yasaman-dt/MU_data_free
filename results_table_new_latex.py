import pandas as pd
from collections import defaultdict


# Load the stats DataFrame
stats_df = pd.read_csv("results_new/results_mean_std_all_numeric.csv")

# Select key columns to display
columns_to_display = [
    ("val_full_retain_acc", r"\mathcal{A}^{all}_r \uparrow"),
    ("val_full_fgt_acc", r"\mathcal{A}^{all}_f\downarrow"),
    #("train_retain_acc", r"\mathcal{A}^{train}_r \uparrow"),
    #("train_fgt_acc", r"\mathcal{A}^{train}_f\downarrow"),
    ("val_test_retain_acc", r"\mathcal{A}^t_r \uparrow"),
    ("val_test_fgt_acc", r"\mathcal{A}^t_f \downarrow"),
    ("AUS", r"AUS \uparrow")
]

# === Helper to determine D_r-free and D_f-free flags
def get_data_free_flags(method, source):
    if method in ["original", "retrained"]:
        return ("--", "--")
    elif method in ["MM"]:
        return (r"\cmark", r"\cmark") 
    elif method in ["FT","RE"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\xmark", r"\cmark")
    elif method in ["NG", "RL", "BS", "BE", "LAU"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\cmark", r"\xmark")
    elif method in ["NGFTW", "DUCK", "SCRUB", "SCAR"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\xmark", r"\xmark")
    return (r"\xmark", r"\xmark")


# Group rows by dataset
datasets = stats_df["dataset"].unique()

# === Define display names and references
method_name_and_ref = {
    "original": ("Original", r"–"),
    "retrained": ("Retrained ", r"–"),
    "RE": ("Retrained FC", r"–"),
    "FT": ("Fine Tuning", r"\citep{golatkar2020eternal} Ours"),
    "NG": ("Neg. Grad.", r"\citep{golatkar2020eternal} Ours"),
    "NGFTW": ("Neg. Grad. +", r"\citep{golatkar2020eternal} Ours"),
    "RL": ("Rand. Lab.", r"\citep{hayase2020selective} Ours"),
    "BS": ("Boundary S.", r"\citep{chen2023boundary} Ours"),
    "BE": ("Boundary E.", r"\citep{chen2023boundary} Ours"),
    "LAU": ("LAU", r"\citep{kim2024layer} Ours"),
    "SCRUB": ("SCRUB", r"\citep{kurmanji2023towards} Ours"),
    "DUCK": ("DUCK", r"\citep{cotogni2023duck} Ours"),
    "SCAR": ("SCAR", r"\citep{bonato2024retain} Ours"),

}



for dataset in datasets:
    dataset_rows = stats_df[stats_df["dataset"] == dataset]

    if dataset == "cifar10":
        dataset_name = "CIFAR10"
    elif dataset == "cifar100":
        dataset_name = "CIFAR100"
    elif dataset == "TinyImageNet":
        dataset_name = "TinyImageNet"
    else:
        continue  # skip unknown dataset
    # Build LaTeX rows
    latex_rows = []
    prev_method_base = None
    
    # Define method order
    method_order = ["original", "retrained", "RE", "FT", "NG", "NGFTW", "RL","BS", "BE", "LAU", "SCRUB", "DUCK", "SCAR"]
    
    # Sort the rows by method_order and then source
    dataset_rows = dataset_rows.copy()
    dataset_rows["method_order"] = dataset_rows["method"].apply(lambda m: method_order.index(m) if m in method_order else len(method_order))
    dataset_rows = dataset_rows.sort_values(by=["method_order", "source"]).drop(columns="method_order")
    
    for idx, (_, row) in enumerate(dataset_rows.iterrows()):
    
        method = row["method"]
        source = row["source"]
        method_base = method  # method name without "(source)"
        method_display_base, default_ref = method_name_and_ref.get(method, (method, r"–"))
        # Override citation for non-original methods based on source
        if method != "original":
            if source == "synth":
                ref = "Ours"
            else:
                ref = default_ref.replace(" Ours", "")  # Keep only the cited work
        else:
            ref = default_ref  # Leave original method as-is

        is_first_of_method = prev_method_base != method_base
        method_display = ""
        ref_display = ""
        
        if is_first_of_method:
            group_rows = dataset_rows[dataset_rows["method"] == method]
            group_count = len(group_rows)
        
            method_display = rf"\multirow{{{group_count}}}{{*}}{{\centering {method_display_base}}}"

    

        if prev_method_base in ["FT", "BE"]:
            latex_rows.append(r"\midrule\midrule")
        else:
            latex_rows.append(r"\midrule")
                    
        
        prev_method_base = method_base
    
        D_r_free, D_f_free = get_data_free_flags(method, source)
        cells = [D_r_free, D_f_free]
    
        for prefix, label in columns_to_display:
            mean_col = f"{prefix}_mean"
            std_col = f"{prefix}_std"
            std_val = (row[std_col]) if pd.notnull(row[std_col]) else 0.0
    
            
            val = row[mean_col]
            std = std_val
            
            if pd.isna(val) or pd.isna(std):
                cell = "-"
            else:
                if label == r"AUS \uparrow":
                    val_str = f"{val:.3f}"
                    std_str = f"{std:.3f}"
                else:
                    val_str = f"{val:.2f}"
                    std_str = f"{std:.2f}"
                    if val < 10: val_str = "0" + val_str
                    if std < 10: std_str = "0" + std_str
            
                cell = f"{val_str} ({std_str})"
    
            cells.append(cell)
    
        latex_rows.append(f"{method_display} & {ref} & " + " & ".join(cells) + r" \\")


    # Format column headers
    access_headers = [
        r"\makecell{$\mathcal{D}_r$\\free}",
        r"\makecell{$\mathcal{D}_f$\\free}"
    ]
    metric_headers = [f"${latex_label}$" for _, latex_label in columns_to_display]
    column_labels = "Reference & " + " & ".join(access_headers + metric_headers)



    # Final LaTeX table with resizebox
    latex_table = fr"""\begin{{table}}[ht]
    \centering
    \caption{{Performance comparison on {dataset_name} using ResNet18. Metrics are shown as mean (std).
              Columns $\mathcal{{D}}_r$ free and $\mathcal{{D}}_f$ free indicate whether the method operates
              without access to the retain set or forget set, respectively,
              denoted by a checkmark (\cmark) for true and a cross (\xmark) for false.}}
    \label{{tab:{dataset.lower()}}}

    \scriptsize
    \resizebox{{\textwidth}}{{!}}{{
    \begin{{tabular}}{{l|c|cc|cc|cc|cc|c}}
    \toprule
    \toprule
    
    Method & {column_labels} \\
    \midrule
    \midrule
    
    """ + "\n".join(latex_rows) + r"""
    \bottomrule
    \bottomrule
    
    \end{tabular}
    }}
    \end{table}
    """

    # Save to file
    filename = f"results_new/table_{dataset.lower()}_new.tex"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"✅ LaTeX table saved to {filename}")




# === Define displayed metrics
columns_to_display = [
    ("val_test_retain_acc", "\mathcal{A}^t_r"),
    ("val_test_fgt_acc", "\mathcal{A}^t_f"),
    ("AUS", "AUS")
]


def sort_key(key):
    base_method = key.split(" (")[0]
    source = key.split(" (")[1].replace(")", "")
    method_idx = method_order.index(base_method) if base_method in method_order else len(method_order)
    source_idx = 0 if source == "real" else 1  # put real before synth
    return (method_idx, source_idx)

# === Group rows by (method, source)
grouped_methods = defaultdict(lambda: {"CIFAR10": ["-"]*3, "CIFAR100": ["-"]*3, "TinyImageNet": ["-"]*3})

access_flags = {}  # Store access flags per (method, source) once

for _, row in stats_df.iterrows():
    method = row["method"]
    source = row["source"]


    dataset = row["dataset"].strip().lower()
    if dataset == "cifar10":
        dataset = "CIFAR10"
    elif dataset == "cifar100":
        dataset = "CIFAR100"
    elif "tiny" in dataset:
        dataset = "TinyImageNet"
    else:
        continue  # skip unknown dataset


    key = f"{method} ({source})"
    values = []

    for prefix, label in columns_to_display:
        mean_col = f"{prefix}_mean"
        std_col = f"{prefix}_std"
        std_val = (row[std_col]) if pd.notnull(row[std_col]) else 0.0
    
        val = row[mean_col]
        std = std_val
        
        if pd.isna(val) or pd.isna(std):
            cell = "-"
        else:
            if label == "AUS":
                val_str = f"{val:.3f}"
                std_str = f"{std:.3f}"
            else:
                val_str = f"{val:.2f}"
                std_str = f"{std:.2f}"
                if val < 10: val_str = "0" + val_str
                if std < 10: std_str = "0" + std_str
        
            cell = f"{val_str} ({std_str})"
    
        cell = f"{val_str} ({std_str})"
        values.append(cell)  # ← THIS WAS MISSING



    grouped_methods[key][dataset] = values
    access_flags[key] = get_data_free_flags(method, source)

# === Build LaTeX table
latex_table = r"""\begin{table}[ht]
\centering
\caption{Performance comparison on CIFAR10, CIFAR100, and TinyImageNet using ResNet-18 as the base architecture.
         For each dataset, we fine-tune three independently initialized models and perform class-wise unlearning separately for every class.
         Reported metrics are the mean and standard deviation computed across all classes and model seeds.
         Columns $\mathcal{D}_r$ free and $\mathcal{D}_f$ free indicate whether the method operates without
         access to the retain or forget set, respectively, with (\cmark) denoting true and (\xmark) denoting false.}


\label{tab:main_results}

\resizebox{\textwidth}{!}{
\begin{tabular}{l|c|cc|ccc|ccc|ccc}
\toprule
\toprule
\multirow{2}{*}{Method} & \multirow{2}{*}{Reference} & \multirow{2}{*}{\shortstack{$\mathcal{D}_r$ \\ free}} & \multirow{2}{*}{\shortstack{$\mathcal{D}_f$ \\ free}} & \multicolumn{3}{c|}{\textbf{CIFAR10}} & \multicolumn{3}{c|}{\textbf{CIFAR100}} & \multicolumn{3}{c}{\textbf{TinyImageNet}} \\
 &  &  &  & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS \uparrow & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS \uparrow & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS \uparrow\\
\midrule
\midrule
"""






# Sort by method name for consistency

prev_base_method = None
method_counts = defaultdict(int)

# Count how many times each method appears
for key in grouped_methods.keys():
    base_method = key.split(" (")[0]
    method_counts[base_method] += 1

for idx, key in enumerate(sorted(grouped_methods.keys(), key=sort_key)):
    base_method = key.split(" (")[0]

    if prev_base_method and base_method != prev_base_method:
        latex_table += r"\midrule" + "\n"
        if prev_base_method in ["FT", "BE"]:
            latex_table += r"\midrule" + "\n"

        
    D_r_free, D_f_free = access_flags[key]
    values = grouped_methods[key]["CIFAR10"] + grouped_methods[key]["CIFAR100"] + grouped_methods[key]["TinyImageNet"]

    # Get display name and citation
    method_display_base, default_ref = method_name_and_ref.get(base_method, (base_method, r"–"))
    
    # Recover source (real/synth) from key
    source = key.split(" (")[1].replace(")", "")

    if base_method != "original":
        if source == "synth":
            ref = "Ours"
        else:
            ref = default_ref.replace(" Ours", "")  # Keep only the cited work
    else:
        ref = default_ref  # Leave original method as-is



    if method_counts[base_method] > 1:
        if source == "real":
            method_cell = rf"\multirow{{{method_counts[base_method]}}}{{*}}{{\centering {method_display_base}}}"
        else:
            method_cell = ""
        ref_cell = ref
    else:
        method_cell = method_display_base
        ref_cell = ref


    row = [method_cell, ref_cell, D_r_free, D_f_free] + values

    latex_table += " & ".join(row) + r" \\" + "\n"

    prev_base_method = base_method
    
    

# Close LaTeX
latex_table += r"""\bottomrule
\bottomrule
\end{tabular}%
}
\end{table}
"""

# === Save to file (UTF-8)
with open("results_new/table_total_new.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("✅ LaTeX table saved to combined_table.tex")





