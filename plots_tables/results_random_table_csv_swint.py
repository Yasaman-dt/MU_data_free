import os
import pandas as pd
import re
from glob import glob
import numpy as np


# === Setup paths ===
parent_dir = r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_head_swint"
sources = [
    ("results_real", "real"),
    ("results_synth_gaussian_10000perclass", "synth"),
    ("results_synth_gaussian_seed42", "synth"),
    ("results_synth_gaussian_seed0", "synth"),
    ("results_synth_laplace", "synth"),
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


original_path = os.path.join(parent_dir, "results_real/results_original_swint.csv")

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

original_summary.to_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_head_swint/original_averaged_results_swint.csv", index=False)

metrics = ['val_test_retain_acc', 'val_test_fgt_acc', 'val_full_retain_acc', 'val_full_fgt_acc', 'AUS']

# Compute mean and std
df_original_grouped = original_df.groupby(['dataset', 'model', 'mode', 'Forget Class'])[metrics].agg(['mean', 'std']).reset_index()

# Flatten MultiIndex columns
df_original_grouped.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df_original_grouped.columns]


# Load the uploaded CSV files
cifar10_df = pd.read_csv(f"{parent_dir}/results_real/retrained/cifar10_swint_unlearning_summary.csv")
cifar100_df = pd.read_csv(f"{parent_dir}/results_real/retrained/cifar100_swint_unlearning_summary.csv")
tinyimagenet_df = pd.read_csv(f"{parent_dir}/results_real/retrained/tinyImagenet_swint_unlearning_summary.csv")

# Add dataset identifiers
cifar10_df["dataset"] = "CIFAR10"
cifar100_df["dataset"] = "CIFAR100"
tinyimagenet_df["dataset"] = "TinyImageNet"

# Combine all into one DataFrame
retrained_df =cifar10_df
retrained_df = pd.concat([cifar10_df, cifar100_df, tinyimagenet_df], ignore_index=True)
retrained_df = retrained_df.rename(columns={"class_removed": "Forget Class"})
retrained_df = retrained_df.rename(columns={"best_val_acc": "val_test_retain_acc"})
retrained_df = retrained_df.rename(columns={"train_acc": "train_retain_acc"})

def infer_noise_type(path_or_name: str) -> str:
    s = os.path.normpath(path_or_name).replace("\\", "/").lower()
    if "results_real" in s:
        return "none"
    if "gaussian" in s:
        return "gaussian"
    if "laplace" in s:
        return "laplace"
    if "uniform" in s:
        return "uniform"
    if "sigma" in s:   # treat sigma* folders as gaussian noise families
        return "gaussian"
    return "unknown"


original_df["noise_type"] = "none"
retrained_df["noise_type"] = "none"


# Rename the column 'best_val_acc' to 'val_full_retain_acc'

# Add 'val_full_fgt_acc' column with all values set to 0
retrained_df["val_test_fgt_acc"] = 0.0
retrained_df["train_fgt_acc"] = 0.0
retrained_df["val_full_fgt_acc"] = 0.0

retrained_df['Forget Class'] = pd.to_numeric(retrained_df['Forget Class'], errors='coerce')
original_df['Forget Class']  = pd.to_numeric(original_df['Forget Class'], errors='coerce')

# --- Ensure retrained has model_num by broadcasting over originals ---

# 1) Normalize/ensure model_num on originals
if 'Model Num' in original_df.columns:
    original_df.rename(columns={'Model Num': 'model_num'}, inplace=True)
if 'n_model' in original_df.columns and 'model_num' not in original_df.columns:
    original_df.rename(columns={'n_model': 'model_num'}, inplace=True)
original_df['model_num'] = pd.to_numeric(original_df['model_num'], errors='coerce')

# 2) If retrained already has model_num, keep it.
#    Otherwise, replicate retrained rows across each original model_num per (dataset, Forget Class).
if 'model_num' not in retrained_df.columns or retrained_df['model_num'].isna().all():
    key_cols = ['dataset', 'Forget Class']

    # All original seeds (model_num) available for each (dataset, Forget Class)
    orig_models = (
        original_df[key_cols + ['model_num']]
        .dropna(subset=['model_num'])
        .drop_duplicates()
    )

    # Cartesian/broadcast: one row per (dataset, Forget Class, model_num),
    # copying the single retrained metrics across the original model_num's
    retrained_df = orig_models.merge(retrained_df, on=key_cols, how='left')

    # Optional sanity check: warn if some (dataset, Forget Class) in originals
    # had no matching retrained entry
    missing_pairs = retrained_df[retrained_df['val_test_retain_acc'].isna()]
    if not missing_pairs.empty:
        print("[WARN] Missing retrained rows for these (dataset, Forget Class):")
        print(missing_pairs[key_cols].drop_duplicates().to_string(index=False))

# Ensure numeric type after merge
retrained_df['model_num'] = pd.to_numeric(retrained_df['model_num'], errors='coerce')

key_cols = ['dataset', 'Forget Class', 'model_num']

orig_baseline = (
    original_df[key_cols + ['val_test_retain_acc']]
    .dropna(subset=['val_test_retain_acc'])
    .groupby(key_cols, as_index=False)['val_test_retain_acc']
    .mean()
    .rename(columns={'val_test_retain_acc': 'val_test_retain_acc_orig'})
)

retrained_df = retrained_df.merge(orig_baseline, on=key_cols, how='left')

unmatched = retrained_df[retrained_df['val_test_retain_acc_orig'].isna()]
if not unmatched.empty:
    print("[WARN] No matching original baseline for these rows:")
    print(unmatched[key_cols].drop_duplicates().to_string(index=False))

retrained_df['AUS'] = 1 - (
    (retrained_df['val_test_retain_acc_orig'] - retrained_df['val_test_retain_acc']) / 100.0
)
# retrained_df['AUS'] = retrained_df['AUS'].clip(0, 1)  # optional


# Save the combined DataFrame
output_path = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_head_swint/results_retrained_swint.csv"
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
                # if model_num not in [2, 3, 4]:
                #     continue

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
                df["noise_type"] = infer_noise_type(file_path) 

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
    all_columns = set()
    for df in all_data:
        all_columns.update(df.columns)
    
    # Step 2: Ensure every DataFrame has all columns
    for i in range(len(all_data)):
        for col in all_columns:
            if col not in all_data[i].columns:
                all_data[i][col] = 0  # or np.nan
        
    
    final_df = pd.concat(all_data, ignore_index=True)

    # Save merged raw results
    final_df.to_csv(os.path.join(parent_dir, "results_unlearning_swint.csv"), index=False)
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
    best_df.to_csv(os.path.join(parent_dir, "results_unlearning_best_per_model_by_aus_swint.csv"), index=False)
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


    # for df in [original_df]:
    #     if "method" in df.columns:
    #         df["method"] = df["method"].replace(method_map)
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
    
    save_dir = os.path.join(parent_dir, "best_per_dataset_method_source_swint")
    os.makedirs(save_dir, exist_ok=True)


    for (dataset, method, source), group_df in best_df.groupby(["dataset", "method", "source"]):
        filename = f"{dataset}_{method}_{source}.csv"
        output_file = os.path.join(save_dir, filename)
        group_df.to_csv(output_file, index=False)
        #print(f"✅ Saved {output_file}")    
    
    # === Combine original + best_df
    combined_df = pd.concat([best_df, original_df, retrained_df], ignore_index=True)
    #combined_df = pd.concat([best_df, original_df], ignore_index=True)

    combined_df.to_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_head_swint/results_total_swint.csv", index=False)


    col = 'val_test_fgt_acc'
    
    if col in combined_df.columns:
        # Exempt original model rows from filtering
        is_original = combined_df.get('method', '').astype(str).str.lower().eq('original')
    
        # Apply >50 filter only to non-original rows
        mask_bad = (~is_original) & combined_df[col].notna() & (combined_df[col] > 20)
    
        removed_rows = combined_df[mask_bad].copy()
        kept_rows = combined_df[~mask_bad].copy()
    
        removed_path = os.path.join(parent_dir, "filtered_out_rows_over50_val_test_fgt_acc_swint.csv")
        kept_path = os.path.join(parent_dir, "filtered_in_rows_over50_val_test_fgt_acc_swint.csv")
        removed_rows.to_csv(removed_path, index=False)
        kept_rows.to_csv(kept_path, index=False)
    
        print(f"🧹 Filtered out {len(removed_rows)} rows with {col} > 50 (excluding original). Kept {len(kept_rows)} rows for stats.")
    else:
        print(f"⚠️ Column '{col}' not found; proceeding without filtering.")

    
    #df_final = combined_df
    df_final = kept_rows


    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols1 = df_final.select_dtypes(include='number').columns
    stats_df1 = df_final.groupby(['Forget Class', "dataset", "method", "model", "source"])[numeric_cols1].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df1.columns = ['_'.join(col).strip('_') for col in stats_df1.columns.values]

    stats_path1 = os.path.join(parent_dir, "mean_std_results_by_class_model_dataset_method_source_swint.csv")
    stats_df1.to_csv(stats_path1, index=False)
    print("✅ Mean and std of all numeric columns saved.")


    print("✅ Merged original results with current best results.")
    
    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols = df_final.select_dtypes(include='number').columns
    stats_df = df_final.groupby(["dataset", "method", "model", "source"])[numeric_cols].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    stats_path = os.path.join(parent_dir, "results_mean_std_all_numeric_swint.csv")
    stats_df.to_csv(stats_path, index=False)
    print("✅ Mean and std of all numeric columns saved.")

else:
    print("❌ No data loaded.")
    
    
