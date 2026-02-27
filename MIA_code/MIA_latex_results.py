import os
import math
import pandas as pd
from decimal import Decimal, ROUND_DOWN
from collections import defaultdict
import re

# ============================================================
# USER CONFIG
# ============================================================
REAL_DIR  = "MIA_results_real"
SYNTH_DIR = "MIA_results_synth"

dataset_to_show = "cifar10"  # e.g. "cifar10", "cifar100", "tinyimagenet"

# canonical model keys used in the table:
model_order   = ["resnet18", "resnet50", "vit", "swint"]
model_display = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "vit": "ViT-B/16", "swint": "Swin-T"}

# metrics you want in the table
MIA1_COL = "confidence"     # -> MIA_I
MIA2_COL = "cv_score_mean"  # -> MIA_II

# method display names + order
method_name_and_ref = {
    "original":  ("Original", "–"),
    "retrained": ("Retrained", "–"),
    "FT":        ("FT \\citep{golatkar2020eternal}", "–"),
    "NG":        ("NG \\citep{golatkar2020eternal}", "–"),
    "NGFTW":     ("NG+ \\citep{kurmanji2023towards}", "–"),
    "RL":        ("RL \\citep{hayase2020selective}", "–"),
    "BS":        ("BS \\citep{chen2023boundary}", "–"),
    #"BE":        ("BE \\citep{chen2023boundary}", "–"),
    "LAU":       ("LAU \\citep{kim2024layer}", "–"),
    "SCRUB":     ("SCRUB \\citep{kurmanji2023towards}", "–"),
    "DUCK":      ("DUCK \\citep{cotogni2023duck}", "–"),
    "SCAR":      ("SCAR \\citep{bonato2024retain}", "–"),
    "DELETE":    ("DELETE \citep{zhou2025decoupled}", "–"),

}
method_order = ["original", "retrained", "FT", "NG", "RL", "BS", "DELETE", "NGFTW", "SCRUB", "SCAR"]

# ============================================================
# HELPERS: normalization + safe aggregation
# ============================================================
def coalesce_columns(df, canonical, candidates):
    """Create canonical column from candidates by taking first non-null across them. Drop extras."""
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return df
    if canonical not in df.columns:
        df[canonical] = None
    # take first non-null value left-to-right
    df[canonical] = df[cols].bfill(axis=1).iloc[:, 0]
    # drop non-canonical duplicates
    drop_cols = [c for c in cols if c != canonical]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def normalize_dataset_name(x):
    if pd.isna(x):
        return x
    s = str(x).strip().lower()
    # common variants
    if s in ["cifar-10", "cifar10", "cifar_10"]:
        return "cifar10"
    if s in ["cifar-100", "cifar100", "cifar_100"]:
        return "cifar100"
    if s in ["tinyimagenet", "tiny_image_net", "tiny-image-net", "tiny_imageNet", "tinyimagenet200", "tiny-imagenet"]:
        return "tinyimagenet"
    return s

def normalize_model_name(x):
    if pd.isna(x):
        return x
    s = str(x).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    # resnet
    if "resnet18" in s:
        return "resnet18"
    if "resnet50" in s:
        return "resnet50"
    # vit variants
    if "vit" in s:
        return "vit"
    if "swint" in s:
        return "swint"
    return s  # fallback

def normalize_method_name(x):
    if pd.isna(x):
        return x
    return str(x).strip()

def normalize_frame(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # coalesce common variants into canonical names
    df = coalesce_columns(df, "ForgetClass", ["Forget Class", "ForgetClass", "forget_class", "forgetclass"])
    df = coalesce_columns(df, "Dataset", ["Dataset", "dataset", "DATASET"])
    df = coalesce_columns(df, "Model",   ["Model", "model", "MODEL", "Backbone", "backbone"])
    df = coalesce_columns(df, "Method",  ["Method", "method", "METHOD"])
    df = coalesce_columns(df, "source",  ["source", "Source", "SOURCE"])

    # normalize key fields
    if "Dataset" in df.columns:
        df["Dataset"] = df["Dataset"].apply(normalize_dataset_name)
    if "Model" in df.columns:
        df["Model"] = df["Model"].apply(normalize_model_name)
    if "Method" in df.columns:
        df["Method"] = df["Method"].apply(normalize_method_name)

    return df

def agg_mean_std_numeric(df, group_cols, metric_allowlist=None):
    """
    Aggregate mean/std over numeric metric columns.
    metric_allowlist: if provided, only these metrics are aggregated.
    """
    df = normalize_frame(df)

    # decide which metric columns to aggregate
    metric_cols = [c for c in df.columns if c not in group_cols]
    if metric_allowlist is not None:
        metric_cols = [c for c in metric_cols if c in metric_allowlist]

    # coerce metrics to numeric (strings -> NaN)
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep numeric + not all NaN
    metric_cols = [
        c for c in metric_cols
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]

    if not metric_cols:
        # return empty but with expected columns
        return pd.DataFrame(columns=group_cols)

    mean_df = df.groupby(group_cols)[metric_cols].mean().reset_index()
    std_df  = df.groupby(group_cols)[metric_cols].std().reset_index()
    return mean_df.merge(std_df, on=group_cols, suffixes=("_mean", "_std"))

# formatting
def truncate_to_2(x):
    return Decimal(str(x)).quantize(Decimal("1.00"), rounding=ROUND_DOWN)

MISSING = "-"   # or r"\text{--}"

def fmt_mean_std_percent(mean_val, std_val):
    if mean_val is None or std_val is None:
        return MISSING
    if isinstance(mean_val, float) and math.isnan(mean_val):
        return MISSING
    if isinstance(std_val, float) and math.isnan(std_val):
        return MISSING
    return f"{truncate_to_2(mean_val)} $\\pm$ {truncate_to_2(std_val)}"

def get_metrics_for(method, source, model_name):
    row = latex_df[(latex_df["Method"] == method) &
                   (latex_df["source"] == source) &
                   (latex_df["Model"] == model_name)]
    if row.empty:
        return (MISSING, MISSING)

    r = row.iloc[0]

    mia1_mean = r.get(f"{MIA1_COL}_mean", float("nan")) * 100
    mia1_std  = r.get(f"{MIA1_COL}_std",  float("nan")) * 100
    mia2_mean = r.get(f"{MIA2_COL}_mean", float("nan")) * 100
    mia2_std  = r.get(f"{MIA2_COL}_std",  float("nan")) * 100

    mia1_str = fmt_mean_std_percent(mia1_mean, mia1_std)
    mia2_str = fmt_mean_std_percent(mia2_mean, mia2_std)
    return (mia1_str, mia2_str)


# your data-free flags logic
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

# ============================================================
# LOAD CSVs (no overwriting)
# ============================================================
def load_folder_csvs(folder_path, source_label):
    out = []
    for fn in os.listdir(folder_path):
        if not fn.endswith(".csv"):
            continue
        fp = os.path.join(folder_path, fn)
        df = pd.read_csv(fp, index_col=False)
        df["source"] = source_label
        df["_filename"] = fn
        out.append(df)
    return out

real_dfs  = load_folder_csvs(REAL_DIR,  "real")
synth_dfs = load_folder_csvs(SYNTH_DIR, "synth")
all_dfs = real_dfs + synth_dfs

if not all_dfs:
    raise RuntimeError("No CSV files found in MIA_results_real/ or MIA_results_synth/")

def has_tag(df, tag):
    # df["_filename"] exists but may be empty => unique() could be empty
    if "_filename" not in df.columns:
        return False
    u = df["_filename"].unique()
    return len(u) > 0 and (tag in u[0])

forget_dfs = [d for d in all_dfs if has_tag(d, "MIA_forget_efficacy")]
mia2_dfs   = [d for d in all_dfs if has_tag(d, "MIA2_forget_efficacy")]
mia3_dfs   = [d for d in all_dfs if has_tag(d, "MIA3_efficacy")]
priv_dfs   = [d for d in all_dfs if has_tag(d, "training_privacy")]


# concat
df_forget  = pd.concat(forget_dfs,  ignore_index=True) if forget_dfs  else pd.DataFrame()
df_mia2    = pd.concat(mia2_dfs,    ignore_index=True) if mia2_dfs    else pd.DataFrame()
df_mia3    = pd.concat(mia3_dfs,    ignore_index=True) if mia3_dfs    else pd.DataFrame()
df_privacy = pd.concat(priv_dfs, ignore_index=True) if priv_dfs else pd.DataFrame()

# ============================================================
# AGGREGATE (ONLY NUMERIC METRICS)
# ============================================================
group_cols = ["source", "Dataset", "Model", "Method"]

# only aggregate the two metrics we need for the table
agg_forget = agg_mean_std_numeric(df_forget, group_cols, metric_allowlist=[MIA1_COL])
agg_mia2   = agg_mean_std_numeric(df_mia2,   group_cols, metric_allowlist=[MIA2_COL])

# merge into one frame for latex
latex_df = agg_forget.merge(agg_mia2, on=group_cols, how="left")

# filter dataset
latex_df = normalize_frame(latex_df)
latex_df = latex_df[latex_df["Dataset"] == normalize_dataset_name(dataset_to_show)].copy()

# quick sanity prints (optional)
print("Datasets found:", sorted(latex_df["Dataset"].dropna().unique().tolist()))
print("Models found:", sorted(latex_df["Model"].dropna().unique().tolist()))
print("Methods found:", sorted(latex_df["Method"].dropna().unique().tolist()))

# ============================================================
# BUILD LaTeX TABLE: backbones side-by-side with MIA_I / MIA_II
# ============================================================


# Count rows per method (real/synth) for multirow
method_counts = defaultdict(int)
for m in method_order:
    for src in ["real", "synth"]:
        if not latex_df[(latex_df["Method"] == m) & (latex_df["source"] == src)].empty:
            method_counts[m] += 1

n_models = len(model_order)
n_models = len(model_order)
col_spec = "c|cc|" + "|".join(["cc"] * n_models)

latex_lines = [
    r"\begin{table*}[ht]",
    r"\centering",
    r"\captionsetup{font=small}",
    rf"\caption{{MIA performance of single-class unlearning on CIFAR10 using ResNet-18, ResNet-50, ViT-B/16 and Swin-T, averaged over 5 random trials. Rows highlighted in gray represent our results using synthetic embeddings, while the corresponding non-shaded rows use original embeddings with the same method.}}",
    r"\label{tab:MIA_results_backbones}",
    r"\resizebox{\textwidth}{!}{",
    rf"\begin{{tabular}}{{{col_spec}}}",
    r"\toprule",
    r"\toprule",
]

# --- Header row 1: Method + Dr/Df + model groups (like your second example) ---
header1 = [
    r"\multirow{2}{*}{Method}",
    r"\multirow{2}{*}{\shortstack{$\mathcal{D}_r$\\free}}",
    r"\multirow{2}{*}{\shortstack{$\mathcal{D}_f$\\free}}",
]

for i, mk in enumerate(model_order):
    disp = model_display.get(mk, mk)
    header1.append(
        rf"\multicolumn{{2}}{{c{'|' if i < len(model_order)-1 else ''}}}{{\textbf{{{disp}}}}}"
    )

latex_lines.append(" & ".join(header1) + r" \\")



# --- Header row 2: only the metrics; first 3 cells are "covered" by the multirow ---
header2 = ["", "", ""]
for _ in model_order:
    header2 += [r"$\text{MIA}_{I}\uparrow$", r"$\text{MIA}_{II}\downarrow$"]
latex_lines.append(" & ".join(header2) + r" \\")

latex_lines.append(r"\midrule")
latex_lines.append(r"\midrule")

prev_base_method = None

for method in method_order:
    rows_for_method = []

    for source in ["real", "synth"]:
        # Skip if method/source doesn't exist in this dataset for any model
        exists_any = any(
            not latex_df[(latex_df["Method"] == method) &
                         (latex_df["source"] == source) &
                         (latex_df["Model"] == mdl)].empty
            for mdl in model_order
        )
        if not exists_any:
            continue

        dr_free, df_free = get_data_free_flags(method, source)

        metric_cells = []
        for mdl in model_order:
            mia1_str, mia2_str = get_metrics_for(method, source, mdl)
            metric_cells += [mia1_str, mia2_str]

        rows_for_method.append((dr_free, df_free, metric_cells, source))

    if prev_base_method is not None and method != prev_base_method:
        if prev_base_method in ["original", "FT", "DELETE"]:
            latex_lines.append(r"\midrule")
            latex_lines.append(r"\midrule")
        else:
            latex_lines.append(r"\midrule")

    for i, (dr_free, df_free, metric_cells, source) in enumerate(rows_for_method):
        # multirow method name if both real+synth exist
        if method_counts[method] > 1:
            if i == 0:
                method_cell = rf"\multirow{{{method_counts[method]}}}{{*}}{{{method_name_and_ref.get(method, (method,))[0]}}}"
            else:
                method_cell = ""
        else:
            method_cell = method_name_and_ref.get(method, (method,))[0]

        row = [method_cell, dr_free, df_free] + metric_cells

        # Gray shading for synth row (excluding method cell)
        if source == "synth":
            row = [row[0]] + [rf"\cellcolor{{gray!15}}{cell}" for cell in row[1:]]

        latex_lines.append(" & ".join(row) + r" \\")

    prev_base_method = method

latex_lines += [
    r"\bottomrule",
    r"\bottomrule",
    r"\end{tabular}}",
    r"\end{table*}",
]

out_tex = "MIA_backbones_table.tex"
with open(out_tex, "w") as f:
    f.write("\n".join(latex_lines))

print(f"Wrote: {out_tex}")

