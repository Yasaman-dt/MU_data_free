import os
import pandas as pd
import re
from glob import glob

# === Setup paths ===
parent_dir = r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"
sources = ["results_real", "results_synth"]
original_path = os.path.join(parent_dir, "results/results_original.csv")

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

# Group by fixed Dataset, Model, and Model Num, and compute mean and std
original_summary = original_df.groupby(['dataset', 'model', 'Model Num'])[metrics].agg(['mean', 'std'])

# Flatten the MultiIndex columns for better readability
original_summary.columns = ['_'.join(col).strip() for col in original_summary.columns.values]



original_summary = original_summary.reset_index()

original_summary.to_csv("results/original_averaged_results.csv", index=False)

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
output_path = "results/results_retrained.csv"
retrained_df.to_csv(output_path, index=False)



all_data = []

for source in sources:
    base_dir = os.path.join(parent_dir, source)
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

                df = pd.read_excel(file_path) if filename.endswith(".xlsx") else pd.read_csv(file_path)
                df["dataset"] = dataset
                df["model"] = model
                df["model_num"] = model_num
                df["lr"] = lr_value
                df["method"] = method
                df["source"] = "real" if source == "results_real" else "synth"

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
    final_df.to_csv(os.path.join(parent_dir, "results/results_unlearning.csv"), index=False)
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
    best_df.to_csv(os.path.join(parent_dir, "results/results_unlearning_best_per_model_by_aus.csv"), index=False)
    print("✅ Refined best results saved using AUS → val_test_fgt_acc → val_test_retain_acc.")


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
    

    # (Optional) Add missing columns if needed
    for col in best_df.columns:
        if col not in original_df.columns:
            original_df[col] = None  # Fill with NaN
        if col not in retrained_df.columns:
            retrained_df[col] = None
        
    # Align column order
    original_df = original_df[best_df.columns]
    retrained_df = retrained_df[best_df.columns]
    
    save_dir = os.path.join(parent_dir, "results/best_per_dataset_method_source")
    os.makedirs(save_dir, exist_ok=True)


    for (dataset, method, source), group_df in best_df.groupby(["dataset", "method", "source"]):
        filename = f"{dataset}_{method}_{source}.csv"
        output_file = os.path.join(save_dir, filename)
        group_df.to_csv(output_file, index=False)
        print(f"✅ Saved {output_file}")    
    
    
    # === Combine original + best_df
    combined_df = pd.concat([best_df, original_df, retrained_df], ignore_index=True)
    combined_df.to_csv("results/results_total.csv", index=False)

    print("✅ Merged original results with current best results.")
    
    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols = combined_df.select_dtypes(include='number').columns
    stats_df = combined_df.groupby(["dataset", "method", "model", "source"])[numeric_cols].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    stats_path = os.path.join(parent_dir, "results/results_mean_std_all_numeric.csv")
    stats_df.to_csv(stats_path, index=False)
    print("✅ Mean and std of all numeric columns saved.")

else:
    print("❌ No data loaded.")
