import os
import pandas as pd
import re
from glob import glob
import numpy as np


# === Setup paths ===
parent_dir = r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"
sources = [
    ("results_fc_resnet18/results_real", None, "real"),
    ("results_fc_resnet18/sigma0.5_persamplefix/results_synth", 0.5, "synth"),
    ("results_fc_resnet18/sigma5.0_persamplefix/results_synth", 5.0, "synth"),
    ("results_fc_resnet18/sigma0.0_persamplefix/results_synth", 0.0, "synth"),
    ("results_fc_resnet18/sigma0.0_persamplefix_gaussian/results_synth", 0.0, "synth"),

]


method_map = {
    "FineTuning": "FT",
    "BoundaryShrink": "BS",
    "BoundaryExpanding": "BE",
    "RandomLabels": "RL",
    "RetrainedEmbedding": "RE",
    "NegativeGradient": "NG",
    "NGFT_weighted": "NGFTW",
    "DELETE": "DELETE",
}


original_path = os.path.join(parent_dir, "results_fc_resnet18/results_real/results_original_resnet18.csv")

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

original_summary.to_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_fc_resnet18/original_averaged_results_resnet18.csv", index=False)

metrics = ['val_test_retain_acc', 'val_test_fgt_acc', 'val_full_retain_acc', 'val_full_fgt_acc', 'AUS']

# Compute mean and std
df_original_grouped = original_df.groupby(['dataset', 'model', 'mode', 'Forget Class'])[metrics].agg(['mean', 'std']).reset_index()

# Flatten MultiIndex columns
df_original_grouped.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df_original_grouped.columns]



# Load the uploaded CSV files
cifar10_df = pd.read_csv(f"{parent_dir}/results_fc_resnet18/results_real/retrained/cifar10_resnet18_unlearning_summary.csv")
cifar100_df = pd.read_csv(f"{parent_dir}/results_fc_resnet18/results_real/retrained/cifar100_resnet18_unlearning_summary.csv")
tinyimagenet_df = pd.read_csv(f"{parent_dir}/results_fc_resnet18/results_real/retrained/tinyImagenet_resnet18_unlearning_summary.csv")

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
output_path = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_fc_resnet18/results_retrained_resnet18.csv"
retrained_df.to_csv(output_path, index=False)

original_df["sigma"] = None
retrained_df["sigma"] = None

all_data = []

for folder_name, sigma, source_type in sources:
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
                df["sigma"] = sigma

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
    final_df.to_csv(os.path.join(parent_dir, "results_fc_resnet18/results_unlearning_resnet18.csv"), index=False)
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
    best_df.to_csv(os.path.join(parent_dir, "results_fc_resnet18/results_unlearning_best_per_model_by_aus_resnet18.csv"), index=False)
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
    
    save_dir = os.path.join(parent_dir, "results_fc_resnet18/best_per_dataset_method_source_resnet18")
    os.makedirs(save_dir, exist_ok=True)


    for (dataset, method, source), group_df in best_df.groupby(["dataset", "method", "source"]):
        filename = f"{dataset}_{method}_{source}.csv"
        output_file = os.path.join(save_dir, filename)
        group_df.to_csv(output_file, index=False)
        #print(f"✅ Saved {output_file}")    
    
    # === Combine original + best_df
    combined_df = pd.concat([best_df, original_df, retrained_df], ignore_index=True)
    combined_df.to_csv("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/results_fc_resnet18/results_total_resnet18.csv", index=False)


    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols1 = combined_df.select_dtypes(include='number').columns
    stats_df1 = combined_df.groupby(['Forget Class', "dataset", "method", "model", "source"])[numeric_cols1].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df1.columns = ['_'.join(col).strip('_') for col in stats_df1.columns.values]

    stats_path1 = os.path.join(parent_dir, "results_fc_resnet18/mean_std_results_by_class_model_dataset_method_source_resnet18.csv")
    stats_df1.to_csv(stats_path1, index=False)
    print("✅ Mean and std of all numeric columns saved.")


    print("✅ Merged original results with current best results.")
    
    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols = combined_df.select_dtypes(include='number').columns
    stats_df = combined_df.groupby(["dataset", "method", "model", "source"])[numeric_cols].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    stats_path = os.path.join(parent_dir, "results_fc_resnet18/results_mean_std_all_numeric_resnet18.csv")
    stats_df.to_csv(stats_path, index=False)
    print("✅ Mean and std of all numeric columns saved.")

else:
    print("❌ No data loaded.")
    
    
