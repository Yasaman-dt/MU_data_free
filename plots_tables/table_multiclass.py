import os
import re
from glob import glob

import numpy as np
import pandas as pd

# ============================================================
#  Paths and configuration
# ============================================================

parent_dir = r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"

# Folders that contain the per-method subfolders
sources = [
    ("multi_class/results_real/samples_per_class_500", None, "real"),
    ("multi_class/results_synth_gaussian/samples_per_class_500", None, "synth"),
]

# How to rename methods for the final tables
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


# === Define display names and references (same style as your big table) ===
method_name_and_ref = {
    "original": ("Original", "–"),
    "retrained": (r"\makecell{Retrained}", "–"),
    "RE":        (r"\makecell{Retrained (FC)}", "–"),
    "FT": ("FT \\citep{golatkar2020eternal}", "–"),
    "NG": ("NG \\citep{golatkar2020eternal}", "–"),
    "NGFTW": ("NG+ \\citep{kurmanji2023towards}", "–"),
    "RL": ("RL \\citep{hayase2020selective}", "–"),
    "BS": ("BS \\citep{chen2023boundary}", "–"),
    "BE": ("BE \\citep{chen2023boundary}", "–"),
    "LAU": ("LAU \\citep{kim2024layer}", "–"),
    "SCRUB": ("SCRUB \\citep{kurmanji2023towards}", "–"),
    "DUCK": ("DUCK \\citep{cotogni2023duck}", "–"),
    "SCAR": ("SCAR \\citep{bonato2024retain}", "–"),
    "DELETE": ("DELETE \\citep{zhou2025decoupled}", "–"),
}


original_path = os.path.join(parent_dir, "multi_class/results_real/samples_per_class_500/results_original_resnet18.csv")

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




# === D_r-free / D_f-free flags (same logic as your FC table) ===
def get_data_free_flags(method, source):
    if method in ["original", "retrained"]:
        return ("--", "--")
    elif method in ["MM"]:
        return (r"\cmark", r"\cmark")
    elif method in ["FT", "RE"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\xmark", r"\cmark")
    elif method in ["NG", "RL", "BS", "BE", "LAU", "DELETE"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\cmark", r"\xmark")
    elif method in ["NGFTW", "DUCK", "SCRUB", "SCAR"]:
        return (r"\cmark", r"\cmark") if source == "synth" else (r"\xmark", r"\xmark")
    return (r"\xmark", r"\xmark")


# ------------------------------------------------------------
#  Helper: infer noise type from path
# ------------------------------------------------------------
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
    if "sigma" in s:  # treat sigma* folders as gaussian noise families
        return "gaussian"
    return "unknown"

def make_red(text: str) -> str:
    if text in ("", "-"):
        return text
    return rf"\textcolor{{red}}{{{text}}}"

# ------------------------------------------------------------
#  Helper: format mean ± std for LaTeX
# ------------------------------------------------------------
def fmt_pm(row: pd.Series, metric: str) -> str:
    """Return 'mean ± std' LaTeX string for a metric in stats_df."""
    m = row.get(f"{metric}_mean", np.nan)
    s = row.get(f"{metric}_std", np.nan)
    if pd.isna(m) or pd.isna(s):
        return "-"

    # 3 decimals for AUS, 2 for all accuracy metrics
    if metric == "AUS":
        val = f"{m:.3f} $\\pm$ {s:.3f}"
    else:
        val = f"{m:.2f} $\\pm$ {s:.2f}"

    return make_red(val)


def count_forget(s):
    """
    This function counts how many classes are in a composite `Forget Class`.
    The composite `Forget Class` values are split by underscores, and each part represents a class.
    Specifically checks for 2, 5, and 10 class counts.
    """
    if pd.isna(s) or str(s).strip() == "" or str(s).lower() == "none":
        return 0
    # Split by underscores, count the number of class identifiers
    class_count = len(str(s).split("_"))
    
    # Ensure classes 2, 5, and 10 are properly represented in the table
    if class_count in [2, 5, 10]:
        return class_count
    return 0  # Ignore other class counts for the table

# Adjust LaTeX table generation for proper handling of `Forget Class`
def make_multi_forget_table(stats_df: pd.DataFrame, out_path: str, dataset: str = "cifar100", model: str = "resnet18", forget_list=(2, 5, 10)):
    """
    stats_df: aggregated mean/std by (dataset, method, model, source, num_forget)
    Writes a LaTeX table with column groups for each num_forget in forget_list.
    First columns: Method, D_r-free, D_f-free.
    """
    method_order = ["original", "FT", "NG", "RL","BE", "DELETE", "NGFTW", "RE"]  # Including "original"

    method_display = {m: method_name_and_ref.get(m, (m, "–"))[0] for m in method_order}

    metrics = ["val_test_retain_acc", "val_test_fgt_acc", "AUS"]
    metric_header = {
        "val_test_retain_acc": r"$\mathcal{A}^t_r \uparrow$",
        "val_test_fgt_acc": r"$\mathcal{A}^t_f \downarrow$",
        "AUS": r"AUS $\uparrow$",
    }

    sub = stats_df[(stats_df["dataset"] == dataset) & (stats_df["model"] == model)]

    n_cols_group = len(metrics)
    total_metric_cols = len(forget_list) * n_cols_group

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    caption_text = (
        r"Unlearning performance on CIFAR-100 using ResNet-18 as the base architecture. "
        r"Rows highlighted in gray correspond to methods applied on synthetic embeddings, "
        r"while the non-shaded rows use original embeddings. "
        r"Columns $\mathcal{D}_r$-free and $\mathcal{D}_f$-free indicate whether the method operates "
        r"without access to the retain or forget set, respectively, with (\cmark) indicating data-free "
        r"operation and (\xmark) indicating that the corresponding data is required."
    )

    lines.append(rf"\caption{{{make_red(caption_text)}}}")
    lines.append(r"\label{tab:ResNet-18_cifar100_multi_class}")
    lines.append(r"\resizebox{0.99\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{c|cc|ccc|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(r"\toprule")

    # First header row: Method / D_r-free / D_f-free span two rows
    header1 = r"\multirow{2}{*}{Method} & \multirow{2}{*}{\makecell{$\mathcal{D}_r$\\free}} & \multirow{2}{*}{\makecell{$\mathcal{D}_f$\\free}}"
    for nf in forget_list:
        header1 += rf" & \multicolumn{{{n_cols_group}}}{{c}}{{\textbf{{{nf}-Classes}}}}"
    header1 += r" \\"
    lines.append(header1)

    # Second header row: metric names under each class group
    cells = ["", "", ""]  # Starting columns for Method, D_r-free, D_f-free
    for _nf in forget_list:
        cells.extend(metric_header[m] for m in metrics)

    header2 = " & ".join(cells) + r"\\"
    lines.append(header2)
    lines.append(r"\midrule")
    lines.append(r"\midrule")

    # Generate rows for each method (real and synth)
    for m in method_order[0:]:
        df_m = sub[sub["method"] == m]
        if df_m.empty:
            continue

        df_m = df_m.copy()  # Explicitly create a copy of the dataframe
        df_m.loc[:, "num_forget"] = df_m["Forget Class"].apply(count_forget)

        real_cells_all = []
        synth_cells_all = []
        for nf in forget_list:
            df_nf = df_m[df_m["num_forget"] == nf]

            # real row
            real_row = df_nf[df_nf["source"] == "real"]
            if not real_row.empty:
                r = real_row.iloc[0]
                real_cells_all.extend([fmt_pm(r, met) for met in metrics])
            else:
                real_cells_all.extend(["-"] * len(metrics))

            # synth row
            synth_row = df_nf[df_nf["source"] == "synth"]
            if not synth_row.empty:
                s = synth_row.iloc[0]
                synth_cells_all.extend([fmt_pm(s, met) for met in metrics])
            else:
                synth_cells_all.extend(["-"] * len(metrics))

        method_tex = method_display.get(m, m)

        rows_for_method = []
        if any(cell != "-" for cell in real_cells_all):
            rows_for_method.append(("real", real_cells_all))
        if any(cell != "-" for cell in synth_cells_all):
            rows_for_method.append(("synth", synth_cells_all))

        if not rows_for_method:
            continue

        n_rows = len(rows_for_method)
        for idx, (source, cells_src) in enumerate(rows_for_method):
            if idx == 0:
                method_cell = rf"\multirow{{{n_rows}}}{{*}}{{{method_tex}}}"
            else:
                method_cell = ""

            D_r_free, D_f_free = get_data_free_flags(m, source)

            # Build the full row as a list of cells
            row = [method_cell, D_r_free, D_f_free] + cells_src

            # 👉 Shade synthetic rows in gray (except the Method cell)
            if source == "synth":
                row = [row[0]] + [
                    rf"\cellcolor{{gray!15}}{cell}" if cell != "" else ""
                    for cell in row[1:]
                ]

            line = " & ".join(row) + r" \\"
            lines.append(line)

        lines.append(r"\midrule")
        if m in ["original", "FT", "DELETE"]:
            lines.append(r"\midrule")



    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_table = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(latex_table)

    print(f"✅ LaTeX multi-forget table saved to: {out_path}")


# ============================================================
#  MAIN PIPELINE
# ============================================================

def main():
    # --------------------------------------------------------
    # 3) Load all unlearning experiment CSVs (real + synthetic)
    # --------------------------------------------------------
    all_data = []

    for folder_name, sigma, source_type in sources:
        base_dir = os.path.join(parent_dir, folder_name)
        if not os.path.isdir(base_dir):
            continue

        methods = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]

        for method in methods:
            method_path = os.path.join(base_dir, method)
            file_pattern = os.path.join(method_path, "*_unlearning_summary_m*_lr*")
            files = glob(file_pattern)

            for file_path in files:
                filename = os.path.basename(file_path)

                match = re.match(
                    r"(?P<dataset>[^_]+)_(?P<model>[^_]+)_unlearning_summary_m(?P<model_num>\d+)_lr(?P<lr>[\d\.]+)",
                    filename,
                )

                if not match:
                    print(f"⚠️ Could not parse: {filename}")
                    continue

                dataset = match.group("dataset")
                model = match.group("model")
                model_num = int(match.group("model_num"))
                lr_value = float(match.group("lr").rstrip("."))

                try:
                    df = (
                        pd.read_excel(file_path)
                        if filename.endswith(".xlsx")
                        else pd.read_csv(file_path)
                    )
                except pd.errors.ParserError as e:
                    print(f"❌ Parser error in file: {file_path}")
                    print(str(e))
                    continue

                df["dataset"] = dataset
                df["model"] = model
                df["model_num"] = model_num
                df["lr"] = lr_value
                df["method"] = method_map.get(method, method)
                df["source"] = source_type
                df["sigma"] = sigma
                df["noise_type"] = infer_noise_type(file_path)

                # convert accuracies to %
                acc_cols = [
                    "train_retain_acc",
                    "train_fgt_acc",
                    "val_test_retain_acc",
                    "val_test_fgt_acc",
                    "val_full_retain_acc",
                    "val_full_fgt_acc",
                ]
                for col in acc_cols:
                    if col in df.columns:
                        df[col] = df[col] * 100.0

                all_data.append(df)

    if not all_data:
        print("❌ No unlearning data loaded.")
        return

    # unify all columns
    all_columns = set()
    for df in all_data:
        all_columns.update(df.columns)
    for i in range(len(all_data)):
        for col in all_columns:
            if col not in all_data[i].columns:
                all_data[i][col] = 0

    final_df = pd.concat(all_data, ignore_index=True)
    out_merged = os.path.join(parent_dir, "multi_class/results_unlearning_resnet18.csv")
    final_df.to_csv(out_merged, index=False)
    print("✅ All unlearning results merged:", out_merged)

    # --------------------------------------------------------
    # 4) Choose best config per (source, method, dataset, model,
    #    model_num, Forget Class) using AUS → ...
    # --------------------------------------------------------
    sort_keys = [
        "AUS",
        "val_test_fgt_acc",
        "val_test_retain_acc",
        "val_full_fgt_acc",
        "val_full_retain_acc",
    ]
    ascending_flags = [False, True, False, True, False]

    sorted_df = final_df.sort_values(by=sort_keys, ascending=ascending_flags)

    best_df = (
        sorted_df.groupby(
            ["source", "method", "dataset", "model", "model_num", "Forget Class"],
            as_index=False,
        )
        .first()
        .copy()
    )

    out_best = os.path.join(
        parent_dir, "multi_class/results_unlearning_best_per_model_by_aus_resnet18.csv"
    )
    best_df.to_csv(out_best, index=False)
    print("✅ Best configs per model saved:", out_best)

    # --------------------------------------------------------
    # 5) Save per-(dataset, method, source) CSVs (optional)
    # --------------------------------------------------------
    save_dir = os.path.join(
        parent_dir, "multi_class/best_per_dataset_method_source_resnet18"
    )
    os.makedirs(save_dir, exist_ok=True)

    for (dataset, method, source), group_df in best_df.groupby(
        ["dataset", "method", "source"]
    ):
        filename = f"{dataset}_{method}_{source}.csv"
        output_file = os.path.join(save_dir, filename)
        group_df.to_csv(output_file, index=False)

    original_df["method"] = "original"
    original_df["source"] = "real"
    original_df["dataset"] = original_df["dataset"].replace({
    "CIFAR10": "cifar10",
    "CIFAR100": "cifar100"
    })
    # combine everything (here only best_df)
    combined_df = pd.concat([original_df, best_df], ignore_index=True)

    out_total = os.path.join(
        parent_dir, "multi_class/results_total_resnet18.csv"
    )
    combined_df.to_csv(out_total, index=False)
    print("✅ Combined results saved:", out_total)

    # --------------------------------------------------------
    # 6) Add num_forget and compute mean/std (named aggregations)
    # --------------------------------------------------------
    def count_forget(s):
        if pd.isna(s) or str(s).strip() == "" or str(s).lower() == "none":
            return 0
        return len(str(s).split("_"))

    combined_df["num_forget"] = combined_df["Forget Class"].apply(count_forget)

    # === Compute mean and std for all numeric columns, grouped by dataset/method/model/source
    numeric_cols = combined_df.select_dtypes(include='number').columns
    stats_df = combined_df.groupby(['Forget Class', "dataset", "method", "model", "source"])[numeric_cols].agg(['mean', 'std']).reset_index()

    # Flatten multi-level column names
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    stats_path = os.path.join(parent_dir, "multi_class/results_mean_std_all_numeric_resnet18.csv")
    stats_df.to_csv(stats_path, index=False)

    print("✅ Mean/std by dataset/method/model/source/num_forget saved:", stats_path)

    # --------------------------------------------------------
    # 7) Generate ONE LaTeX table (2, 5, 10 forgotten classes)
    # --------------------------------------------------------
    table_out = os.path.join(
        parent_dir,
        "multi_class/table_cifar100_resnet18_multiclass.tex",
    )
    make_multi_forget_table(
        stats_df,
        table_out,
        dataset="cifar100",  # change if you want cifar10, etc.
        model="resnet18",
        forget_list=(2, 5, 10),
    )


if __name__ == "__main__":
    main()
