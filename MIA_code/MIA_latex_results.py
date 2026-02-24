import os
import pandas as pd

folders = {
    'MIA_results_real': 'real',
    'MIA_results_synth': 'synth'
}



# Loop through each folder
for folder_name, label in folders.items():
    folder_path = os.path.join(folder_name)
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # Load CSV
            df = pd.read_csv(file_path)

            # Add source column
            df['source'] = label

            # Overwrite the file with the new column
            df.to_csv(file_path, index=False)
            print(f"Updated {file_path} with source = '{label}'")

folders = ['MIA_results_real', 'MIA_results_synth']


all_forget_dfs = []
all_privacy_dfs = []
all_MIA2_dfs = []
all_MIA3_dfs = []

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path)
            if 'MIA_forget_efficacy' in filename:
                all_forget_dfs.append(df)
            elif 'training_privacy' in filename:
                all_privacy_dfs.append(df)
            elif 'MIA2_forget_efficacy' in filename:
                all_MIA2_dfs.append(df)
            elif 'MIA3_efficacy' in filename:
                all_MIA3_dfs.append(df)                
                
# Combine
df_forget = pd.concat(all_forget_dfs, ignore_index=True)
df_privacy = pd.concat(all_privacy_dfs, ignore_index=True)
df_MIA2 = pd.concat(all_MIA2_dfs, ignore_index=True)
df_MIA3 = pd.concat(all_MIA3_dfs, ignore_index=True)

# Grouping config
group_cols = ['source', 'Dataset', 'Model', 'Method']
value_cols_forget = [col for col in df_forget.columns if col not in group_cols]
value_cols_privacy = [col for col in df_privacy.columns if col not in group_cols]
value_cols_MIA2 = ['cv_score_mean']
value_cols_MIA3 = ['F1_mean']

# Compute mean and std, then merge with suffixes
agg_forget_mean = df_forget.groupby(group_cols)[value_cols_forget].mean().reset_index()
agg_forget_std = df_forget.groupby(group_cols)[value_cols_forget].std().reset_index()
agg_forget = pd.merge(agg_forget_mean, agg_forget_std, on=group_cols, suffixes=('_mean', '_std'))

agg_privacy_mean = df_privacy.groupby(group_cols)[value_cols_privacy].mean().reset_index()
agg_privacy_std = df_privacy.groupby(group_cols)[value_cols_privacy].std().reset_index()
agg_privacy = pd.merge(agg_privacy_mean, agg_privacy_std, on=group_cols, suffixes=('_mean', '_std'))

agg_MIA2_mean = df_MIA2.groupby(group_cols)[value_cols_MIA2].mean().reset_index()
agg_MIA2_std = df_MIA2.groupby(group_cols)[value_cols_MIA2].std().reset_index()
agg_MIA2 = pd.merge(agg_MIA2_mean, agg_MIA2_std, on=group_cols, suffixes=('_mean', '_std'))

agg_MIA3_mean = df_MIA3.groupby(group_cols)[value_cols_MIA3].mean().reset_index()
agg_MIA3_std = df_MIA3.groupby(group_cols)[value_cols_MIA3].std().reset_index()
agg_MIA3 = pd.merge(agg_MIA3_mean, agg_MIA3_std, on=group_cols, suffixes=('_mean', '_std'))


# Save
agg_forget.to_csv('aggregated_forget_efficacy_with_std.csv', index=False)
agg_privacy.to_csv('aggregated_training_privacy_with_std.csv', index=False)
agg_MIA2.to_csv('aggregated_MIA2_with_std.csv', index=False)
agg_MIA3.to_csv('aggregated_MIA3_with_std.csv', index=False)


latex_df = pd.merge(
    pd.merge(
        agg_forget,
        agg_MIA2,
        on=['source', 'Dataset', 'Model', 'Method'],
        how='left'
    ),
    agg_MIA3,
    on=['source', 'Dataset', 'Model', 'Method'],
    how='left'
)

# === Define display names and references
method_name_and_ref = {
    "original": ("Original", "–"),
    "retrained": ("Retrained", "–"),
    #"RE":        (r"\begin{tabular}{c}Retrained \\ (FC)\end{tabular}", "–"),
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

}

# === Order
method_order = ["original", "retrained", "FT", "NG", "RL","BS", "BE", "NGFTW", "SCRUB", "SCAR"]


# For forget efficacy table
MIA1_confidence = "confidence"
#MIA1_correctness = "correctness"
MIA2 = "cv_score_mean"         
MIA3 = "F1_mean"         

columns_to_include = ["source", "Dataset", "Model", "Method",
                      f"{MIA1_confidence}_mean", f"{MIA1_confidence}_std",
                      #f"{MIA1_correctness}_mean", f"{MIA1_correctness}_std",
                      f"{MIA2}_mean", f"{MIA2}_std",
                      #f"{MIA3}_mean", f"{MIA3}_std"
                      ]
latex_df = latex_df[columns_to_include]


latex_lines = [
    r"\begin{table*}[ht]",
    r"\centering",
    r"\captionsetup{font=small}",
    r"\caption{MIA performance of single-class unlearning on CIFAR10 using ResNet-18, averaged over 5 random trials. Rows highlighted in gray represent our results using synthetic embeddings, while the corresponding non-shaded rows use original embeddings with the same method.}",
    r"\label{tab:MIA_results}",
    r"\resizebox{0.7\textwidth}{!}{",
    r"\begin{tabular}{c|cc|c|c}",
    #r"\begin{tabular}{c|cc|c|c|c|c}",
    r"\toprule",
    r"Method & $\mathcal{D}_r$-free & $\mathcal{D}_f$-free & $\text{MIA}_{I}$ $\uparrow$ & $\text{MIA}_{II}$ $\downarrow$ \\",
    #r"Method & $\mathcal{D}_r$-free & $\mathcal{D}_f$-free & $\text{MIA}_{I}(\text{confidence})$ $\uparrow$ & $\text{MIA}_{I}(\text{correctness})$ $\uparrow$ & $\text{MIA}_{II}$ $\downarrow$ & $\text{MIA}_{III}$ $\downarrow$\\",
    r"\midrule"
    r"\midrule"

]

# === Method formatting
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



from collections import defaultdict

# Count how many times each method appears per base
method_counts = defaultdict(int)
for method in method_order:
    for source in ["real", "synth"]:
        row = latex_df[(latex_df["Method"] == method) & (latex_df["source"] == source)]
        if not row.empty:
            method_counts[method] += 1

prev_base_method = None

from decimal import Decimal, ROUND_DOWN

def truncate_to_2(x):
    return Decimal(str(x)).quantize(Decimal('1.00'), rounding=ROUND_DOWN)


for method in method_order:
    base_method = method
    rows_for_method = []

    for source in ["real", "synth"]:
        row = latex_df[(latex_df["Method"] == method) & (latex_df["source"] == source)]
        if row.empty:
            continue

        r = row.iloc[0]
        dr_free, df_free = get_data_free_flags(method, source)
        mia1_confidence_mean = r[f"{MIA1_confidence}_mean"] * 100
        mia1_confidence_std = r[f"{MIA1_confidence}_std"] * 100
        #mia1_confidence_str = f"{mia1_confidence_mean:.2f} $\\pm$ {mia1_confidence_std:.2f}"

        #mia1_correctness_mean = r[f"{MIA1_correctness}_mean"] * 100
        #mia1_correctness_std = r[f"{MIA1_correctness}_std"] * 100
        #mia1_correctness_str = f"{mia1_correctness_mean:.2f} $\\pm$ {mia1_correctness_std:.2f}"


        mia2_mean = r[f"{MIA2}_mean"] * 100
        mia2_std = r[f"{MIA2}_std"] * 100
        #mia2_str = f"{mia2_mean:.2f} $\\pm$ {mia2_std:.2f}"

        #mia3_mean = r[f"{MIA3}_mean"] * 100
        #mia3_std = r[f"{MIA3}_std"] * 100
        #mia3_str = f"{mia3_mean:.2f} $\\pm$ {mia3_std:.2f}"
        
        mia1_confidence_str = f"{truncate_to_2(mia1_confidence_mean)} $\\pm$ {truncate_to_2(mia1_confidence_std)}"
        #mia1_correctness_str = f"{truncate_to_2(mia1_correctness_mean)} $\\pm$ {truncate_to_2(mia1_correctness_std)}"
        mia2_str = f"{truncate_to_2(mia2_mean)} $\\pm$ {truncate_to_2(mia2_std)}"
        #mia3_str = f"{truncate_to_2(mia3_mean)} $\\pm$ {truncate_to_2(mia3_std)}"
        
        
        
        rows_for_method.append((dr_free, df_free, mia1_confidence_str, mia2_str, source))
        #rows_for_method.append((dr_free, df_free, mia1_confidence_str, mia1_correctness_str, mia2_str, mia3_str, source))

    # Insert midrule if base method changed
    if base_method != prev_base_method:
        if prev_base_method in ["original", "FT", "BE"]:
            latex_lines.append(r"\midrule")
            latex_lines.append(r"\midrule")
        else:
            latex_lines.append(r"\midrule")


    #for i, (dr_free, df_free, mia1_confidence_str, mia1_correctness_str, mia2_str, mia3_str, source) in enumerate(rows_for_method):
    for i, (dr_free, df_free, mia1_confidence_str, mia2_str, source) in enumerate(rows_for_method):


        
        if method_counts[method] > 1:
            if i == 0:
                method_cell = rf"\multirow{{{method_counts[method]}}}{{*}}{{{method_name_and_ref.get(method, (method,))[0]}}}"
            else:
                method_cell = ""

        else:
            method_cell = method_name_and_ref.get(method, (method,))[0]


       
        #row = [method_cell, dr_free, df_free, mia1_confidence_str, mia1_correctness_str, mia2_str, mia3_str]
        row = [method_cell, dr_free, df_free, mia1_confidence_str, mia2_str]

        # Apply gray background for synth rows (but NOT the method column)
        if source == "synth":
            row = [row[0]] + [rf"\cellcolor{{gray!15}}{cell}" for cell in row[1:]]

        latex_lines.append(" & ".join(row) + r" \\")

    prev_base_method = base_method


latex_lines += [
    r"\bottomrule",
    r"\end{tabular}}",
    r"\end{table*}"
]

# Save LaTeX
with open("MIA_forget_efficacy_table.tex", "w") as f:
    f.write("\n".join(latex_lines))