import pandas as pd
from collections import defaultdict, Counter

# Load both ResNet-18 and ResNet-50 result files
ViT_df = pd.read_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_head_ViT/results_mean_std_all_numeric_ViT.csv")
swint_df = pd.read_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_head_swint/results_mean_std_all_numeric_swint.csv")

# Add architecture column
ViT_df["arch"] = "ViT"
swint_df["arch"] = "swint"

# Combine the data
stats_df = pd.concat([ViT_df, swint_df], ignore_index=True)


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
    "retrained": (r"\makecell{Retrained}", "–"),
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


columns_to_display = [
    ("val_test_retain_acc", "\mathcal{A}^t_r"),
    ("val_test_fgt_acc", "\mathcal{A}^t_f"),
    ("AUS", "AUS")
]

def sort_key(key):
    # Expect format: "method (source) [arch]"
    method_part = key.split(" (")[0]
    source_part = key.split(" (")[1].split(")")[0]
    arch_part = key.split("[")[-1].replace("]", "")
    
    method_idx = method_order.index(method_part) if method_part in method_order else len(method_order)
    source_idx = 0 if source_part == "real" else 1
    arch_idx = 0 if arch_part == "ViT" else 1  
    return (arch_idx, method_idx, source_idx)


# === Group rows by (method, source)
grouped_methods = defaultdict(lambda: {"CIFAR10": ["-"]*3, "CIFAR100": ["-"]*3, "TinyImageNet": ["-"]*3})

access_flags = {}  # Store access flags per (method, source) once

max_min_tracker = defaultdict(lambda: defaultdict(dict))  # max_min_tracker[arch][dataset][label]

exclude_mask = (
    ((stats_df["arch"] == "ViT") & (stats_df["method"] == "SCRUB")) |
    ((stats_df["arch"] == "swint") & (stats_df["method"] == "RL")) |
    ((stats_df["arch"] == "swint") & (stats_df["method"] == "BS")) |
    ((stats_df["arch"] == "swint") & (stats_df["method"] == "DELETE"))
)
stats_df = stats_df[~exclude_mask].copy()

for arch in ["ViT", "swint"]:
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
            if "retain" in metric or metric == "AUS":  # higher is better
                max_min_tracker[arch][dataset][label] = df_filtered[metric_mean].max()
            elif "fgt" in metric:  # lower is better
                max_min_tracker[arch][dataset][label] = df_filtered[metric_mean].min()


for _, row in stats_df.iterrows():
    if row["method"] in ["DUCK", "RE", "FT"]:
        continue  
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

    key = f"{method} ({source}) [{row['arch']}]"

    values = []

    for prefix, label in columns_to_display:
        mean_col = f"{prefix}_mean"
        std_col = f"{prefix}_std"
        std_val = (row[std_col]) if pd.notnull(row[std_col]) else 0.0
    
        val = row[mean_col]
        std = std_val
    
        arch = row["arch"]
        
        if pd.isna(val) or pd.isna(std):
            cell = "-"
        else:
            if label == "AUS":
                val_str = f"{val:.3f}"
                std_str = f"{std:.3f}"
            if label == "\mathcal{A}^t_r":
                val_str = f"{val:.2f}"
                std_str = f"{std:.2f}"
            if label == "\mathcal{A}^t_f":
                if method == "original":
                    val_str = f"{val:.2f}"
                    std_str = f"{std:.2f}"
                else:
                    val_str = f"{val:.1f}"
                    std_str = f"{std:.1f}"
    
            dset = dataset

            target_val = round(val, 3)
            tracked_val = round(max_min_tracker[arch][dset][label], 3)
    
            # if label in [r"\mathcal{A}^t_r", "AUS"] and target_val == tracked_val:
            #     val_str = f"\\textbf{{{val_str}}}"
            
            cell = f"{val_str}\\scriptsize{{\\,$\\pm$\\,{std_str}}}"
        
        values.append(cell)

    grouped_methods[key][dataset] = values
    access_flags[key] = get_data_free_flags(method, source)
    grouped_methods[key]["arch"] = row["arch"]
    
# === Build LaTeX table
latex_table = r"""\begin{table*}[ht]
\centering
\captionsetup{font=small}
\caption{Class unlearning performance for CIFAR-10, CIFAR-100, and TinyImageNet using ViT-B-16 and Swin-T as the base architecture.
         Rows highlighted in gray represent our results using synthetic embeddings, while the corresponding non-shaded rows use original embeddings with the same method.
         Columns $\mathcal{D}_r$-free and $\mathcal{D}_f$-free indicate whether the method operates without access to the retain or forget set, respectively, with (\cmark) denoting true and (\xmark) denoting false.}
\label{tab:main_results_head_transformer}

\resizebox{\textwidth}{!}{
\begin{tabular}{c|cc|ccc|ccc|ccc}
\toprule
\toprule
 \multirow{2}{*}{Method} & \multirow{2}{*}{\shortstack{$\mathcal{D}_r$ \\ free}} & \multirow{2}{*}{\shortstack{$\mathcal{D}_f$ \\ free}} & \multicolumn{3}{c|}{\textbf{CIFAR-10}} & \multicolumn{3}{c|}{\textbf{CIFAR-100}} & \multicolumn{3}{c}{\textbf{TinyImageNet}} \\
 &  &  & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS $\uparrow$ & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS $\uparrow$ & $\mathcal{A}_r^t \uparrow$ & $\mathcal{A}_f^t \downarrow$ & AUS $\uparrow$\\
\midrule
\midrule
"""

# Sort by method name for consistency
prev_arch = None

prev_base_method = None
method_counts = defaultdict(int)

# Count how many times each method appears
for key in grouped_methods.keys():
    base_method = key.split(" (")[0]
    arch_part = key.split("[")[-1].replace("]", "")
    method_counts[(base_method, arch_part)] += 1


arch_row_counts = defaultdict(int)
for key in grouped_methods:
    arch_row_counts[grouped_methods[key]["arch"]] += 1
    
    
arch_total_rows = defaultdict(int)
for key in grouped_methods:
    arch = grouped_methods[key]["arch"]
    arch_total_rows[arch] += 1

arch_printed = defaultdict(bool)

insert_ViT_next = False
insert_swint_next = False

all_keys_sorted = sorted(grouped_methods.keys(), key=sort_key)

arch_row_printed = defaultdict(bool)


for idx, key in enumerate(sorted(grouped_methods.keys(), key=sort_key)):
    arch = grouped_methods[key]["arch"]

    base_method = key.split(" (")[0]

    # if prev_arch is not None and arch != prev_arch:
    #     latex_table += r"\midrule" + "\n"        
        
    if prev_arch is None or arch != prev_arch:
        if arch == "ViT":
            latex_table += r"\midrule"
            latex_table += r"\multicolumn{12}{c}{\textbf{ViT-B-16:}} \\" + "\n"
        elif arch == "swint":
            latex_table += r"\midrule"+r"\midrule"+r"\midrule"
            latex_table += r"\multicolumn{12}{c}{\textbf{Swin-T:}} \\" + "\n"
               
    prev_arch = arch
    
    if base_method != prev_base_method:
        if arch== "ViT" and prev_base_method in ["original", "retrained", "DELETE"]:
            latex_table += r"\midrule" + "\n" + r"\midrule" 
        elif arch== "swint" and prev_base_method in ["original", "retrained", "NG"]:
            latex_table += r"\midrule" + "\n" + r"\midrule" 
        else:
            latex_table += r"\midrule" + "\n"

    D_r_free, D_f_free = access_flags[key]
    values = grouped_methods[key]["CIFAR10"] + grouped_methods[key]["CIFAR100"] + grouped_methods[key]["TinyImageNet"]

    # Get display name and citation
    method_display_base, default_ref = method_name_and_ref.get(base_method, (base_method, r"–"))
    
    # Recover source (real/synth) from key
    source = key.split(" (")[1].split(")")[0]


    if base_method != "original":
        if source == "synth":
            ref = "Ours"
        else:
            ref = default_ref.replace(" Ours", "")  # Keep only the cited work
    else:
        ref = default_ref  # Leave original method as-is


    if base_method == "original":
        #arch_cell = rf""
        method_cell = rf"\multirow{{2}}{{*}}{{{method_display_base}}}"
        dr_free = rf"\multirow{{2}}{{*}}{{{D_r_free}}}"
        df_free = rf"\multirow{{2}}{{*}}{{{D_f_free}}}"

        values_multirow = [rf"\multirow{{2}}{{*}}{{{v}}}" for v in values]

        #row = [arch_cell, method_cell, dr_free, df_free] + values_multirow
        row = [method_cell, dr_free, df_free] + values_multirow
        latex_table += " & ".join(row) + r" \\" + "\n"
    
        # Now insert an empty second row for spacing and alignment
        
        row = ["", "", ""] + [""] * len(values)
        #row = ["","", "", ""] + [""] * len(values)
        
        latex_table += " & ".join(row) + r" \\" + "\n" +"\midrule"
        
        continue  # skip rest of loop
        
    method_arch_key = (base_method, grouped_methods[key]["arch"])
    
    if method_counts[method_arch_key] > 1:
        first_key_in_method_group = sorted(
            [k for k in grouped_methods if base_method in k and grouped_methods[k]["arch"] == grouped_methods[key]["arch"]],
            key=sort_key
        )[0]
        if key == first_key_in_method_group:
            method_cell = rf"\multirow{{{method_counts[method_arch_key]}}}{{*}}{{{method_display_base}}}"
        else:
            method_cell = ""
    else:
        method_cell = method_display_base

    if not arch_row_printed[arch]:
        arch_cell = rf"\multirow{{{arch_total_rows[arch]}}}{{*}}{{{arch.replace('resnet', 'ResNet-')}}}"
        arch_row_printed[arch] = True
    else:
        arch_cell = ""

    #row = [arch_cell, method_cell, D_r_free, D_f_free] + values
    row = [method_cell, D_r_free, D_f_free] + values
    prev_base_method = base_method  # Update tracker
    
    
    # Apply gray background for synth rows
    if source == "synth":
        row = [row[0]] + [rf"\cellcolor{{gray!15}}{cell}" for cell in row[1:]]
    
    # Print row once
    latex_table += " & ".join(row) + r" \\" + "\n"
    
    # # === INSERT LABEL AFTER each architecture block ===
    # next_idx = idx + 1
    # if next_idx == len(all_keys_sorted) or grouped_methods[all_keys_sorted[next_idx]]["arch"] != arch:
    #     label = arch.replace("resnet", "ResNet-")
    #     latex_table += r"\midrule" + "\n"
    #     latex_table += rf"\multicolumn{{12}}{{c}}{{\textbf{{{label} $\uparrow$}}}} \\" + "\n"
        
# Close LaTeX
latex_table += r"""\bottomrule
\bottomrule
\end{tabular}%
}
\end{table*}
"""

# === Save to file (UTF-8)
with open("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_fc/table_total_random_head_transformer.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("✅ LaTeX table saved to table_total_random_fc.tex")


