import os
import pandas as pd
import re
from glob import glob

# === Setup paths ===
parent_dir = r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"
sources = [
    ("results_random_layer4_1_conv2/results_real", "real"),
    ("results_random_layer4_1_conv2/results_synth", "synth"),
]


method_map = {
    "FineTuning": "FT",
    "BoundaryShrink": "BS",
    "BoundaryExpanding": "BE",
    "RandomLabels": "RL",
    "RetrainedEmbedding": "RE",
    "NegativeGradient": "NG",
    "NGFT_weighted": "NGFTW",
}


original_path = os.path.join(parent_dir, "results_real/results_original.csv")

original_df = pd.read_csv(original_path)

original_df = original_df.rename(columns={
"Mode": "mode",
"Dataset": "dataset",
"Model": "model",
"Train Retain Acc": "train_retain_acc",
"Train Forget Acc": "train_fgt_acc",
"Val Test Retain Acc": "val_test_retain_acc",
"Val Test Forget Acc": "val_test_fgt_acc",
"Val Full Retain Acc": "val_full_retain_acc",
"Val Full Forget Acc": "val_full_fgt_acc",
})



# Define the metrics for which we want to compute mean and std
metrics = [
    'Train Acc', 'Test Acc', 'train_retain_acc', 'train_fgt_acc',
    'val_test_retain_acc', 'val_test_fgt_acc',
    'val_full_retain_acc', 'val_full_fgt_acc', 'AUS'
]

original_df.rename(columns={"Model Num":"model_num"}, inplace=True)


# Group by fixed Dataset, Model, and Model Num, and compute mean and std
original_summary = original_df.groupby(['dataset', 'model', 'model_num'])[metrics].agg(['mean', 'std'])

# Flatten the MultiIndex columns for better readability
original_summary.columns = ['_'.join(col).strip() for col in original_summary.columns.values]



original_summary = original_summary.reset_index()

original_summary.to_csv("results_random_layer4_1_conv2/original_averaged_results.csv", index=False)

metrics = ['val_test_retain_acc', 'val_test_fgt_acc', 'val_full_retain_acc', 'val_full_fgt_acc', 'AUS']

# Compute mean and std
df_original_grouped = original_df.groupby(['dataset', 'model', 'mode', 'Forget Class'])[metrics].agg(['mean', 'std']).reset_index()

# Flatten MultiIndex columns
df_original_grouped.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df_original_grouped.columns]



# Load the uploaded CSV files
cifar10_df = pd.read_csv(f"{parent_dir}/results_real/retrained/cifar10_resnet18_unlearning_summary.csv")
cifar100_df = pd.read_csv(f"{parent_dir}/results_real/retrained/cifar100_resnet18_unlearning_summary.csv")
tinyimagenet_df = pd.read_csv(f"{parent_dir}/results_real/retrained/tinyImagenet_resnet18_unlearning_summary.csv")

# Add dataset identifiers
cifar10_df["dataset"] = "CIFAR10"
cifar100_df["dataset"] = "CIFAR100"
tinyimagenet_df["dataset"] = "TinyImageNet"

# Combine all into one DataFrame
retrained_df = pd.concat([cifar10_df, cifar100_df, tinyimagenet_df], ignore_index=True)
retrained_df = retrained_df.rename(columns={"class_removed": "Forget Class"})
retrained_df = retrained_df.rename(columns={"best_val_acc": "val_test_retain_acc"})
retrained_df = retrained_df.rename(columns={"train_acc": "train_retain_acc"})



# Rename the column 'best_val_acc' to 'val_full_retain_acc'

# Add 'val_full_fgt_acc' column with all values set to 0
retrained_df["val_test_fgt_acc"] = 0.0
retrained_df["train_fgt_acc"] = 0.0
retrained_df["val_full_fgt_acc"] = 0.0

val_test_retain_acc_original = original_df['val_test_retain_acc']
val_test_retain_acc_retrained = retrained_df['val_test_retain_acc']

AUS = 1 - ((val_test_retain_acc_original - val_test_retain_acc_retrained)/100)

retrained_df["AUS"] = AUS

# Save the combined DataFrame
output_path = "results_random_layer4_1_conv2/results_retrained.csv"
retrained_df.to_csv(output_path, index=False)

all_data = []

for folder_name, source_type in sources:
    base_dir = os.path.join(parent_dir, folder_name)

    methods = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    for method in methods:
        method_path = os.path.join(base_dir, method)
        
        # Match all files with unlearning summary pattern
        file_pattern = os.path.join(method_path, "*_unlearning_summary_m*_lr*")
        files = glob(file_pattern)

        for file_path in files:
            filename = os.path.basename(file_path)

            # Extract dataset, model, model_num, and lr
            match = re.match(r"(?P<dataset>[^_]+)_(?P<model>[^_]+)_unlearning_summary_m(?P<model_num>\d+)_lr(?P<lr>[\d\.]+)", filename)
            

            if match:

                dataset = match.group("dataset")
                model = match.group("model")
                model_num = int(match.group("model_num"))
                lr_value = float(match.group("lr").rstrip("."))
                #if model_num not in [2, 3, 4]:
                #    continue

                #df = pd.read_excel(file_path) if filename.endswith(".xlsx") else pd.read_csv(file_path)
                try:
                    df = pd.read_excel(file_path) if filename.endswith(".xlsx") else pd.read_csv(file_path)
                except pd.errors.ParserError as e:
                    print(f"❌ Parser error in file: {file_path}")
                    print(str(e))
                    continue                
                
                df["dataset"] = dataset
                df["model"] = model
                df["model_num"] = model_num
                df["lr"] = lr_value
                df["method"] = method_map.get(method, method)  # Use mapped name if available
                df["source"] = source_type


                # Multiply accuracy columns by 100 if they exist
                acc_cols = [
                    "train_retain_acc", "train_fgt_acc",
                    "val_test_retain_acc", "val_test_fgt_acc",
                    "val_full_retain_acc", "val_full_fgt_acc"
                ]
                for col in acc_cols:
                    if col in df.columns:
                        df[col] = df[col] * 100


                all_data.append(df)
            else:
                print(f"⚠️ Could not parse: {filename}")






# === Combine all ===
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    # Save merged raw results
    final_df.to_csv(os.path.join(parent_dir, "results_random_layer4_1_conv2/results_unlearning.csv"), index=False)
    print("✅ All results merged.")

    # === Refined selection: prefer highest AUS, then smallest val_test_fgt_acc, then largest val_test_retain_acc
    sort_keys = ["AUS", "val_test_fgt_acc", "val_test_retain_acc", "val_full_fgt_acc", "val_full_retain_acc"]
    ascending_flags = [False, True, False, True, False]  # Maximize AUS, minimize fgt, maximize retain
    
    # Sort the full DataFrame with all tie-breaker preferences
    sorted_df = final_df.sort_values(by=sort_keys, ascending=ascending_flags)
    
    # Group and pick the first (best) row for each combination
    best_df = sorted_df.groupby(
        ["source", "method", "dataset", "model", "model_num", "Forget Class"],
        as_index=False
    ).first()
    
    # Save results
    best_df.to_csv(os.path.join(parent_dir, "results_random_layer4_1_conv2/results_unlearning_best_per_model_by_aus.csv"), index=False)
    print("✅ Refined best results saved using AUS → val_test_fgt_acc → val_test_retain_acc.")

    #original_df = original_df[original_df["model_num"].isin([2, 3, 4])]


    retrained_df["method"] = "retrained"
    retrained_df["source"] = "real"
    retrained_df["dataset"] = retrained_df["dataset"].replace({
    "CIFAR10": "cifar10",
    "CIFAR100": "cifar100"
    })
    original_df["method"] = "original"
    original_df["source"] = "real"
    original_df["dataset"] = original_df["dataset"].replace({
    "CIFAR10": "cifar10",
    "CIFAR100": "cifar100"
    })


    for df in [original_df, retrained_df]:
        if "method" in df.columns:
            df["method"] = df["method"].replace(method_map)

        
    # (Optional) Add missing columns if needed
    for col in best_df.columns:
        if col not in original_df.columns:
            original_df[col] = None  # Fill with NaN
        if col not in retrained_df.columns:
            retrained_df[col] = None
        
    # Align column order
    original_df = original_df[best_df.columns]
    retrained_df = retrained_df[best_df.columns]
    
    save_dir = os.path.join(parent_dir, "results_random_layer4_1_conv2/best_per_dataset_method_source")
    os.makedirs(save_dir, exist_ok=True)


    for (dataset, method, source), group_df in best_df.groupby(["dataset", "method", "source"]):
        filename = f"{dataset}_{method}_{source}.csv"
        output_file = os.path.join(save_dir, filename)
        group_df.to_csv(output_file, index=False)
        #print(f"✅ Saved {output_file}")    
    
    
    # === Combine original + best_df
    combined_df = pd.concat([best_df, original_df, retrained_df], ignore_index=True)
    combined_df.to_csv("results_random_layer4_1_conv2/results_total.csv", index=False)



    print("✅ Merged original results with current best results.")
    
    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols = combined_df.select_dtypes(include='number').columns
    stats_df = combined_df.groupby(["dataset", "method", "model", "source"])[numeric_cols].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    stats_path = os.path.join(parent_dir, "results_random_layer4_1_conv2/results_mean_std_all_numeric.csv")
    stats_df.to_csv(stats_path, index=False)
    print("✅ Mean and std of all numeric columns saved.")

else:
    print("❌ No data loaded.")


import pandas as pd
from collections import defaultdict


# Load the stats DataFrame
stats_df = pd.read_csv("results_random_layer4_1_conv2/results_mean_std_all_numeric.csv")

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
\caption{Negative Gradient + Method performance on original and synthetic data (CIFAR10, CIFAR100, and TinyImageNet),
         when generating random samples in layer 4 before the last convolution.
         We fine-tune five independently initialized models and perform class-wise unlearning separately for every class.
        Reported metrics are the mean and standard deviation computed across all classes and model seeds.}

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
with open("results_random_layer4_1_conv2/results_ngftw_real_vs_synth.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("✅ LaTeX table saved to combined_table.tex")





