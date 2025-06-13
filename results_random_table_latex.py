import pandas as pd
from collections import defaultdict


# Load the stats DataFrame
stats_df = pd.read_csv("results_random_fc/results_mean_std_all_numeric.csv")

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
    "retrained": (r"\begin{tabular}{c}Retrained \\ (Full)\end{tabular}", r"–"),
    "RE":        (r"\begin{tabular}{c}Retrained \\ (FC)\end{tabular}", r"–"),
    "FT": ("FT", r"\citep{golatkar2020eternal} Ours"),
    "NG": ("NG", r"\citep{golatkar2020eternal} Ours"),
    "NGFTW": ("NG+", r"\citep{golatkar2020eternal} Ours"),
    "RL": ("RL", r"\citep{hayase2020selective} Ours"),
    "BS": ("BS", r"\citep{chen2023boundary} Ours"),
    "BE": ("BE", r"\citep{chen2023boundary} Ours"),
    "LAU": ("LAU", r"\citep{kim2024layer} Ours"),
    "SCRUB": ("SCRUB", r"\citep{kurmanji2023towards} Ours"),
    "DUCK": ("DUCK", r"\citep{cotogni2023duck} Ours"),
    "SCAR": ("SCAR", r"\citep{bonato2024retain} Ours"),

}


method_order = ["original", "retrained", "RE", "FT", "NG", "RL","BS", "BE", "LAU", "NGFTW", "SCRUB", "DUCK", "SCAR"]




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

max_min_tracker = defaultdict(dict)
for dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
    df_filtered = stats_df[(stats_df["dataset"].str.lower().str.contains(dataset.lower())) & (stats_df["method"] != "DUCK")]
    for metric, label in columns_to_display:
        metric_mean = f"{metric}_mean"
        if "retain" in metric:  # higher is better
            max_min_tracker[dataset][label] = df_filtered[metric_mean].max()
        elif "fgt" in metric:  # lower is better
            max_min_tracker[dataset][label] = df_filtered[metric_mean].min()
        elif metric == "AUS":  # higher is better
            max_min_tracker[dataset][label] = df_filtered[metric_mean].max()

for _, row in stats_df.iterrows():
    if row["method"] == "DUCK":
        continue  # Skip DUCK method    
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
        
        
            # Determine which dataset this AUS belongs to (based on column index)
            dataset_idx = len(values)  # 0–2: CIFAR10, 3–5: CIFAR100, 6–8: TinyImageNet
            if dataset_idx < 3:
                dset = "CIFAR10"
            elif dataset_idx < 6:
                dset = "CIFAR100"
            else:
                dset = "TinyImageNet"

            target_val = round(val, 3)
            tracked_val = round(max_min_tracker[dataset][label], 3)
            
            # Apply bold only if it's the max for retain or AUS
            if label in [r"\mathcal{A}^t_r", "AUS"] and target_val == tracked_val:
                val_str = f"\\textbf{{{val_str}}}"
    
            cell = f"{val_str}$\\text{{\\scriptsize \\,$\\pm$\\,{std_str}}}$"

    
        values.append(cell)  


    grouped_methods[key][dataset] = values
    access_flags[key] = get_data_free_flags(method, source)

# === Build LaTeX table
latex_table = r"""\begin{table}[ht]
\centering
\caption{Performance comparison on CIFAR10, CIFAR100, and TinyImageNet using ResNet-18 as the base architecture. 
For each dataset, we fine-tune five independently initialized models and perform class-wise unlearning separately for every class. 
Reported metrics are the mean and standard deviation computed across all classes and model seeds. 
To ensure fair comparison, the number of generated synthetic samples per class is matched to the number of samples in the original training dataset. 
These synthetic samples are generated in the feature space prior to the fully connected/classifier head of the model. 
Columns $\mathcal{D}_r$-free and $\mathcal{D}_f$-free indicate whether the method operates without access to the retain or forget set, respectively, with (\cmark) denoting true and (\xmark) denoting false.}


\label{tab:main_results}

\resizebox{\textwidth}{!}{
\begin{tabular}{c|c|cc|ccc|ccc|ccc}
\toprule
\toprule
\multirow{2}{*}{Method} & \multirow{2}{*}{Ref} & \multirow{2}{*}{\shortstack{$\mathcal{D}_r$ \\ free}} & \multirow{2}{*}{\shortstack{$\mathcal{D}_f$ \\ free}} & \multicolumn{3}{c|}{\textbf{CIFAR10}} & \multicolumn{3}{c|}{\textbf{CIFAR100}} & \multicolumn{3}{c}{\textbf{TinyImageNet}} \\
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

    if base_method == "original":
        method_cell = rf"\multirow{{2}}{{*}}{{\centering {method_display_base}}}"
        ref_cell = rf"\multirow{{2}}{{*}}{{\centering {ref}}}"
        dr_free = rf"\multirow{{2}}{{*}}{{{D_r_free}}}"
        df_free = rf"\multirow{{2}}{{*}}{{{D_f_free}}}"

        values_multirow = [rf"\multirow{{2}}{{*}}{{{v}}}" for v in values]


        row = [method_cell, ref_cell, dr_free, df_free] + values_multirow
        latex_table += " & ".join(row) + r" \\" + "\n"
    
        # Now insert an empty second row for spacing and alignment
        row = ["", "", "", ""] + [""] * len(values)
        latex_table += " & ".join(row) + r" \\" + "\n" +"\midrule"
        
                
        
        continue  # skip rest of loop

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
with open("results_random_fc/table_total_random.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("✅ LaTeX table saved to combined_table.tex")


# Load the uploaded data
df_latex_input = pd.read_csv("results_random_fc/mean_std_results_by_class_model_dataset_method_source.csv")

# Filter only for CIFAR-10 dataset
cifar10_df = df_latex_input[df_latex_input["dataset"] == "cifar10"].copy()

# # Keep only relevant columns
# columns_to_keep = ["Forget Class", "source", "train_retain_acc_mean", "train_retain_acc_std", 
#                    "val_test_retain_acc_mean", "val_test_retain_acc_std", 
#                    "val_test_fgt_acc_mean", "val_test_fgt_acc_std", 
#                    "AUS_mean", "AUS_std"]

#cifar10_df = cifar10_df[columns_to_keep]


#cifar10_df.to_csv("results_random_fc/cifar10_results.csv", index=False)

# === Step 1: Preprocess table ===
# df_filtered = cifar10_df[
#     cifar10_df["dataset"] == "cifar10"
# ][["Forget Class", "method", "source",
#    "val_test_fgt_acc_mean","val_test_fgt_acc_std",
#    "val_test_retain_acc_mean", "val_test_retain_acc_std",
#    "AUS_mean","AUS_std"]]

df_filtered = cifar10_df[cifar10_df["method"] != "DUCK"]

# Add display name (human-readable method name)
df_filtered["Display Name"] = df_filtered["method"].map(lambda m: method_name_and_ref[m][0])




from collections import defaultdict

# Track best value (max or min) for each class and metric
best_per_class = defaultdict(lambda: defaultdict(dict))  # e.g., best_per_class[dataset][metric][class_id]

# Only process for current dataset, e.g., CIFAR10
dataset_name = "cifar10"
df_filtered = cifar10_df[cifar10_df["method"] != "DUCK"]

for prefix, label in columns_to_display:
    metric_mean = f"{prefix}_mean"
    for class_name in range(10):  # For CIFAR-10 classes 0-9
        class_subset = df_filtered[df_filtered["Forget Class"] == class_name]
        values = class_subset[metric_mean].dropna()

        if "retain" in prefix or prefix == "AUS":
            best_value = values.max() if not values.empty else None
        # elif "fgt" in prefix:
        #     best_value = values.min() if not values.empty else None
        else:
            best_value = None

        best_per_class[dataset_name][prefix][class_name] = best_value
        
        
# === Step 2: Build rows with Source and Metric ===
columns_to_display = [
    ("val_test_fgt_acc", r"$\mathcal{A}^t_f \downarrow$"),
    ("val_test_retain_acc", r"$\mathcal{A}^t_r \uparrow$"),
    ("AUS", r"AUS $\uparrow$")
]

records = []
for (method, source), group in df_filtered.groupby(["method", "source"]):
    display_name = method_name_and_ref[method][0]
    is_single_value = method.lower() in ["retrained"]

    for prefix, metric_name in columns_to_display:
        mean_key = f"{prefix}_mean"
        std_key = f"{prefix}_std"
        fallback_key = prefix  # for Retrained, if only value is available

        row = {
            "Method": display_name,
            "Source": source,
            "Metric": metric_name
        }
        ref_citation = method_name_and_ref.get(method, ("", "–"))[1]
        if method.lower() in ["original", "retrained"]:
            ref = "–"
        elif source == "synth":
            ref = "Ours"
        else:
            ref = ref_citation.replace(" Ours", "")
        row["Ref"] = ref

        for _, row_df in group.iterrows():
            forget_class = int(row_df["Forget Class"])
        
            if is_single_value:
                value = row_df.get(fallback_key, row_df.get(mean_key, float("nan")))
                if pd.notna(value):
                    # Check if it's the best for that class
                    is_best = value == best_per_class["cifar10"][prefix][forget_class]
                    if prefix == "AUS":
                        value_fmt = f"{value:.3f}"
                    else:
                        value_fmt = f"{value:.2f}"
                        if value < 10:
                            value_fmt = "0" + value_fmt
                    row[forget_class] = fr"\textbf{{{value_fmt}}}" if is_best else value_fmt
                else:
                    row[forget_class] = "-"
            else:
                mean = row_df.get(mean_key, float("nan"))
                std = row_df.get(std_key, float("nan"))
                
                if pd.notna(mean) and pd.notna(std):
                    is_best = mean == best_per_class["cifar10"][prefix][forget_class]
                    if prefix == "AUS":
                        mean_fmt = f"{mean:.3f}"
                        std_fmt = f"{std:.3f}"
                    else:
                        mean_fmt = f"{mean:.2f}"
                        std_fmt = f"{std:.2f}"
                        if mean < 10:
                            mean_fmt = "0" + mean_fmt
                        if std < 10:
                            std_fmt = "0" + std_fmt
                    
                    if is_best:
                        row[forget_class] = fr"\textbf{{{mean_fmt}}}$\text{{\scriptsize \,$\pm$\,{std_fmt}}}$"
                    else:
                        row[forget_class] = fr"{mean_fmt}$\text{{\scriptsize \,$\pm$\,{std_fmt}}}$"

                else:
                    row[forget_class] = "-"

        records.append(row)




# === Step 3: Create final DataFrame and format to 2 decimal places ===
final_df = pd.DataFrame(records)

for col in range(10):  # forget classes 0 to 9
    if col in final_df.columns:
        final_df = final_df[["Method", "Source", "Ref", "Metric"] + list(range(10))]


method_name_map = {k: v[0] for k, v in method_name_and_ref.items()}

# Step 2: Convert method_order to display names
method_order_display = [method_name_map[m] for m in method_order if m in method_name_map]

# Step 3: Apply CategoricalDtype to enforce order
method_dtype = pd.CategoricalDtype(categories=method_order_display, ordered=True)
final_df["Method"] = final_df["Method"].astype(method_dtype)



# Step 4: Sort
final_df = final_df.sort_values(["Method", "Source", "Metric"]).reset_index(drop=True)

latex_lines = []
prev_method = prev_source = None

SHOW_SOURCE_COL = False

# Convert to LaTeX lines manually
header = ["Method", "Source", "Ref", "Metric"] + list(range(10))
column_format = "c|c|c|c|" + "c" * 10

# --- pre-compute how many rows each Method will occupy (needed for \multirow) ---
method_row_counts = final_df.groupby("Method", observed=False).size().to_dict()

latex = []
latex.append(r"\resizebox{\textwidth}{!}{%")
latex.append(r"\begin{tabular}{" + column_format + "}")
latex.append(r"\toprule")
latex.append(r"& & & \multicolumn{10}{c}{Forget Class} \\")
# Original header with Method | Source | Metric | 0..9
header_cells = header[:4] + [fr"\multicolumn{{1}}{{c}}{{{class_name}}}" for class_name in header[4:]]
latex.append(" & ".join(header_cells) + r" \\")
latex.append(r"\midrule")

method_source_row_counts = df_filtered.groupby(["method", "source"]).size().to_dict()

prev_method = None
prev_source_key = None
prev_ref_key = None

for i, row in final_df.iterrows():
    method = row["Method"]
    source = row["Source"]
    
    base_method = [k for k, v in method_name_and_ref.items() if v[0] == method]
    base_method = base_method[0] if base_method else method.lower()
    ref_text = method_name_and_ref.get(base_method, ("", "–"))[1]
    if base_method != "original":
        if source == "synth":
            ref = "Ours"
        else:
            ref = ref_text.replace(" Ours", "")
    else:
        ref = ref_text
    
    # Handle proper source display for 'original' and 'retrained'
    source_display = "-" if method in ["Original", "Retrained"] else source
    source_key = (method, source_display)    
        
    
    
    source_display = "-" if method in ["Original", "Retrained"] else source
    source_key = (method, source_display)  # unique per display value

    if prev_method is not None and method != prev_method:
        latex.append(r"\midrule")

    if prev_source_key is not None and source_key != prev_source_key and method == prev_method:
        latex.append(r"\cmidrule(lr){2-" + str(len(header)) + r"}")

    cells = []
    
    if method == prev_method:
        cells.append("")
    else:
        n_rows = method_row_counts[method]
        cells.append(fr"\multirow{{{n_rows}}}{{*}}{{{method}}}")
    
    if source_key == prev_source_key:
        cells.append("")
    else:
        n_rows = method_source_row_counts.get((method.lower(), source), 1)
        cells.append(fr"\multirow{{{n_rows}}}{{*}}{{\centering {source_display}}}")
    
    
    # Add reference (no merging)
    ref_key = (method, source_display)
    
    if ref_key == prev_ref_key:
        cells.append("")
    else:
        cells.append(fr"\multirow[c]{{{n_rows}}}{{*}}{{\centering\arraybackslash {row['Ref']}}}")
    

    
    prev_ref_key = ref_key    
    cells.append(row["Metric"])
    cells.extend([row.get(col, "") for col in header[4:]])

    latex.append(" & ".join(map(str, cells)) + r" \\")
    prev_method = method
    prev_source_key = source_key


latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"}")  # closing resizebox

with open("results_random_fc/cifar10_unlearning_table_per_class.tex", "w") as f:
    f.write("\n".join(latex))