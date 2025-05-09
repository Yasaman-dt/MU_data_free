import os
import pandas as pd
import re
from glob import glob
import matplotlib.pyplot as plt

# === Setup paths ===
parent_dir = r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/"
sources = ["results_n_samples/results_synth"]
original_path = os.path.join(parent_dir, "results/results_original.csv")

original_df = pd.read_csv(original_path)

original_df = original_df.rename(columns={
"Mode": "mode",
"Dataset": "dataset",
"Model Num": "model_num",
"Model": "model",
"Train Retain Acc": "train_retain_acc",
"Train Forget Acc": "train_fgt_acc",
"Val Test Retain Acc": "val_test_retain_acc",
"Val Test Forget Acc": "val_test_fgt_acc",
"Val Full Retain Acc": "val_full_retain_acc",
"Val Full Forget Acc": "val_full_fgt_acc",
})

original_df["source"] = "real"  # Assuming original_df corresponds to real data
original_df["samples_per_class"] = -1  # Or set to a consistent dummy value if not applicable
original_df["method"] = "original"
original_df["epoch"] = 0
original_df["train_retain_acc"] = 0
original_df["train_fgt_acc"] = 0





# Define the metrics for which we want to compute mean and variance
metrics = [
    'Train Acc', 'Test Acc', 'train_retain_acc', 'train_fgt_acc',
    'val_test_retain_acc', 'val_test_fgt_acc',
    'val_full_retain_acc', 'val_full_fgt_acc', 'AUS'
]

method_map = {
    "FT": "FineTuning",
    "BE": "BoundaryExpanding",
    "RL": "RandomLabels",
}

all_data = []




for source in sources:
    base_dir = os.path.join(parent_dir, source)
    
    # Loop over samples_per_class_* directories
    spc_dirs = [d for d in os.listdir(base_dir) if d.startswith("samples_per_class_")]

    for spc_dir in spc_dirs:
        spc_path = os.path.join(base_dir, spc_dir)
        
        # Extract numeric value from 'samples_per_class_10'
        match_spc = re.match(r"samples_per_class_(\d+)", spc_dir)
        if not match_spc:
            print(f"⚠️ Could not extract samples_per_class from {spc_dir}")
            continue
        
        samples_per_class = int(match_spc.group(1))

        # Now get all method subdirectories under this path
        methods = [name for name in os.listdir(spc_path) if os.path.isdir(os.path.join(spc_path, name))]

        for method in methods:
            method_path = os.path.join(spc_path, method)
            
            file_pattern = os.path.join(method_path, "*_unlearning_summary_m*_lr*")
            files = glob(file_pattern)

            for file_path in files:
                filename = os.path.basename(file_path)
                match = re.match(r"(?P<dataset>[^_]+)_(?P<model>[^_]+)_unlearning_summary_m(?P<model_num>\d+)_lr(?P<lr>[\d\.]+)", filename)
                if match:
                    dataset = match.group("dataset")
                    model = match.group("model")
                    model_num = int(match.group("model_num"))
                    lr_value = float(match.group("lr").rstrip("."))

                    try:
                        df = pd.read_excel(file_path) if filename.endswith(".xlsx") else pd.read_csv(file_path, on_bad_lines='skip')
                    except Exception as e:
                        print(f"❌ Failed to read {file_path}: {e}")
                        continue
                    df["dataset"] = dataset
                    df["model"] = model
                    df["model_num"] = model_num
                    df["lr"] = lr_value
                    df["method"] = method_map.get(method, method)  # Replace if in map, else keep original
                    df["source"] = "real" if source == "results_real" else "synth"
                    df["samples_per_class"] = samples_per_class


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

def normalize_keys(df):
    df['dataset'] = df['dataset'].astype(str).str.strip().str.lower()
    df['model'] = df['model'].astype(str).str.strip().str.lower()
    df['model_num'] = df['model_num'].astype(int)  # ensure consistent type
    df['Forget Class'] = df['Forget Class'].astype(int)
    return df



# === Combine all ===
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    # Save merged raw results
    final_df.to_csv(os.path.join(parent_dir, f"results_n_samples/results_unlearning.csv"), index=False)
    print("✅ All results merged.")

    final_df = final_df[final_df['model_num'].between(2,4)]  # This filters the data

    # === Refined selection: prefer highest AUS, then smallest val_test_fgt_acc, then largest val_test_retain_acc
    sort_keys = ["AUS", "val_test_fgt_acc", "val_test_retain_acc", "val_full_fgt_acc", "val_full_retain_acc"]
    ascending_flags = [False, True, False, True, False]  # Maximize AUS, minimize fgt, maximize retain
    
    # Sort the full DataFrame with all tie-breaker preferences
    sorted_df = final_df.sort_values(by=sort_keys, ascending=ascending_flags)


    # Group and pick the first (best) row for each combination
    best_df = sorted_df.groupby(
        ["source", "method", "dataset", "model", "model_num", "Forget Class", "samples_per_class"],
        as_index=False
    ).first()
    
    best_df = normalize_keys(best_df)
    original_df = normalize_keys(original_df)

    # List of columns you want to bring from original_df with _orig suffix
    cols_to_add = [
        'train_retain_acc', 'train_fgt_acc',
        'val_test_retain_acc', 'val_test_fgt_acc',
        'val_full_retain_acc', 'val_full_fgt_acc',
        'AUS'
    ]
    
    # Set merge keys (these identify the row identity)
    merge_keys = ["dataset", "model", "model_num", "Forget Class"]

    original_subset = original_df[merge_keys + cols_to_add].copy()
    original_subset = original_subset.rename(columns={col: f"{col}_orig" for col in cols_to_add})
    merged_df = best_df.merge(original_subset, on=merge_keys, how='left')
    merged_df['key'] = merged_df['AUS'] > merged_df['AUS_orig']
    
    merged_df['AUS_new'] = merged_df['key'] * merged_df['AUS'] + (1- merged_df['key']) * merged_df['AUS_orig'] 
    merged_df['train_fgt_acc_new'] = merged_df['key'] * merged_df['val_full_fgt_acc'] + (1- merged_df['key']) * merged_df['val_full_fgt_acc_orig'] 
    merged_df['train_retain_acc_new'] = merged_df['key'] * merged_df['val_full_retain_acc'] + (1- merged_df['key']) * merged_df['val_full_retain_acc_orig'] 
    merged_df['val_test_fgt_acc_new'] = merged_df['key'] * merged_df['val_test_fgt_acc'] + (1- merged_df['key']) * merged_df['val_test_fgt_acc_orig'] 
    merged_df['val_test_retain_acc_new'] = merged_df['key'] * merged_df['val_test_retain_acc'] + (1- merged_df['key']) * merged_df['val_test_retain_acc_orig'] 
    merged_df['val_full_fgt_acc_new'] = merged_df['key'] * merged_df['val_full_fgt_acc'] + (1- merged_df['key']) * merged_df['val_full_fgt_acc_orig'] 
    merged_df['val_full_retain_acc_new'] = merged_df['key'] * merged_df['val_full_retain_acc'] + (1- merged_df['key']) * merged_df['val_full_retain_acc_orig'] 

    metrics = [
        'train_retain_acc', 'train_fgt_acc',
        'val_test_retain_acc', 'val_test_fgt_acc',
        'val_full_retain_acc', 'val_full_fgt_acc', 'AUS'
    ]
    
    # Construct list of columns to keep
    columns_to_keep = [col for col in merged_df.columns if not any(
        col == m or col.endswith('_orig') and m in col for m in metrics
    )]
    
    # Create new DataFrame
    new_df = merged_df[columns_to_keep].copy()

    new_df = new_df.rename(columns=lambda col: col.replace('_new', '') if col.endswith('_new') else col)

    best_df = new_df
    # Save results
    best_df.to_csv(os.path.join(parent_dir, "results_n_samples/results_unlearning_best_per_model_by_aus.csv"), index=False)
    print("✅ Refined best results saved using AUS → val_test_fgt_acc → val_test_retain_acc.")



    # === Save one file per (dataset, method, source) ===
    save_dir = os.path.join(parent_dir, "results_n_samples/best_per_dataset_method_source")
    os.makedirs(save_dir, exist_ok=True)


    os.makedirs(os.path.join(save_dir, "samples_per_class"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "forget_class"), exist_ok=True)

    for (dataset, method, source, samples_per_class), group_df in best_df.groupby(["dataset", "method", "source", "samples_per_class"]):
        filename = f"samples_per_class/{dataset}_{method}_{source}_{samples_per_class}.csv"
        output_file = os.path.join(save_dir, filename)
        group_df.to_csv(output_file, index=False)
        #print(f"✅ Saved {output_file}")

        
    for (dataset, method, source, forget_class), group_df in best_df.groupby(["dataset", "method", "source", "Forget Class"]):
        filename = f"forget_class/{dataset}_{method}_{source}_{forget_class}.csv"
        output_file = os.path.join(save_dir, filename)
        group_df.to_csv(output_file, index=False)
        #print(f"✅ Saved {output_file}")

    
    # === Combine original + best_df
    combined_df = pd.concat([best_df], ignore_index=True)
    combined_df.to_csv("results_n_samples/results_total.csv", index=False)

    print("✅ Merged original results with current best results.")
    
    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols1 = combined_df.select_dtypes(include='number').columns
    stats_df1 = combined_df.groupby(["dataset", "method", "model", "source", "samples_per_class"])[numeric_cols1].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df1.columns = ['_'.join(col).strip('_') for col in stats_df1.columns.values]

    stats_path1 = os.path.join(parent_dir, "results_n_samples/results_mean_variance_for_fix_samples_per_class.csv")
    stats_df1.to_csv(stats_path1, index=False)
    print("✅ Mean and std of all numeric columns saved.")


    numeric_cols2 = combined_df.select_dtypes(include='number').columns
    stats_df2 = combined_df.groupby(["dataset", "method", "model", "source", "samples_per_class", "Forget Class"])[numeric_cols2].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df2.columns = ['_'.join(col).strip('_') for col in stats_df2.columns.values]

    stats_path2 = os.path.join(parent_dir, "results_n_samples/results_mean_variance_for_fix_samples_per_class_&_forget_Class.csv")
    stats_df2.to_csv(stats_path2, index=False)
    print("✅ Mean and std of all numeric columns saved.")
    
    
    
else:
    print("❌ No data loaded.")

