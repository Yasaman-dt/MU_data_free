import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.models as models
from create_embeddings_utils import get_model
import matplotlib.pyplot as plt
import os
import seaborn as sns


DIR = "/projets/Zdehghani/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"
model_name = "resnet18"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = {
    #"cifar10": 10,
    #"cifar100": 100,
    "TinyImageNet": 200,
}

# List of model indices to run
model_indices = [1, 2, 3, 4, 5]  # ← Add as many model indices as you want
noise_types = ["gaussian", "bernoulli", "uniform", "laplace", "gumbel"]

# ------------------ Noise Sampling ------------------
def sample_noise(noise_type, size, device='cuda'):
    if noise_type == "gaussian":
        return torch.randn(size, device=device)
    elif noise_type == "bernoulli":
        return torch.bernoulli(torch.full(size, 0.5, device=device))
    elif noise_type == "uniform":
        return torch.empty(size, device=device).uniform_(-1, 1)
    elif noise_type == "laplace":
        return torch.distributions.Laplace(0.0, 1.0).sample(size).to(device)
    elif noise_type == "gumbel":
        return torch.distributions.Gumbel(0.0, 1.0).sample(size).to(device)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

# ------------------ Analysis Function ------------------
def analyze_class_probabilities_from_noise(fc_layer, embedding_dim, device, num_samples, noise_type):
    size = (num_samples, embedding_dim)
    feature_samples = sample_noise(noise_type, size, device)
    with torch.no_grad():
        logits = fc_layer(feature_samples)
        preds = torch.argmax(logits, dim=1)
    class_counts = torch.bincount(preds, minlength=num_classes).cpu().numpy()
    class_probs = class_counts / num_samples

    return class_probs

# ---------- Loop over datasets ----------
for dataset_name, num_classes in datasets.items():
    print(f"Processing {dataset_name.upper()}")
    records = []
    if dataset_name.lower() in ["cifar10", "cifar100"]:
        dataset_name_upper = dataset_name.upper()
    else:
        dataset_name_upper = dataset_name  # e.g., "TinyImageNet"
        
        
    for n_model in model_indices:
        checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"
        model = get_model(model_name, dataset_name_upper, num_classes, checkpoint_path=checkpoint_path_model)
        model.eval()

        fc_layer = model.fc.to(device)
        bbone = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        embedding_dim = bbone(dummy_input).shape[1]

        for noise in noise_types:
            class_probs = analyze_class_probabilities_from_noise(fc_layer, embedding_dim, device, 1000000, noise)
            for i, prob in enumerate(class_probs):
                records.append({
                    "Dataset": dataset_name,
                    "Model": n_model,
                    "Noise Type": noise,
                    "Class": i,
                    "Proportion": prob
                })

    all_results_df = pd.DataFrame(records)
    output_csv_path = os.path.join(DIR, f"results_diff_sampling/all_class_probabilities.csv")
    
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        all_results_df = pd.concat([existing_df, all_results_df], ignore_index=True)

    # Define epsilon (you can change this value as needed)
    epsilon = 0.01

    # Calculate N for each proportion
    all_results_df['N_min'] = np.log(epsilon) / np.log(1 - all_results_df['Proportion'])
    all_results_df['N'] = np.ceil(all_results_df['N_min']).astype(int)
        
    
    all_results_df.to_csv(output_csv_path, index=False)
    print(f"Saved all results to: {output_csv_path}")





# import seaborn as sns
# import matplotlib.pyplot as plt

# # Filter for CIFAR10 only
# subset = all_results_df[all_results_df["Dataset"] == "cifar10"]

# plt.figure(figsize=(16, 7))
# sns.barplot(
#     data=subset,
#     x="Class", y="Proportion", hue="Noise Type",
#     errorbar="sd"
# )
# plt.title("Mean Proportion per Class with Std Dev for Each Noise Type (CIFAR10)")
# plt.xlabel("Class")
# plt.ylabel("Mean Proportion")
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.savefig("barplot_class_distribution_cifar10.png")  # Optional: save the plot
# plt.show()


# import pandas as pd
# import math 

# filepath = 'results_diff_sampling/all_class_probabilities.csv'
# df = pd.read_csv(filepath)

df = all_results_df



grouped_df = df.groupby(["Dataset", "Model", "Noise Type"]).agg(
    N_mean=("N", "mean"),
    N_std=("N", "std"),
    N_min=("N", "min"),
    N_max=("N", "max")
).reset_index()

# Save the grouped statistics to a new CSV file with 2 decimal places
output_filepath = 'results_diff_sampling/avg_var_with_Nmin_Nmax.csv'
grouped_df.to_csv(output_filepath, index=False, float_format='%.2f')


df = grouped_df

#df = pd.read_csv("results_diff_sampling//avg_var_with_Nmin_Nmax.csv")

df = df[df["Model"] == 1]

# Define custom dataset order
dataset_order = ["cifar10", "cifar100", "TinyImageNet"]
df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
df["Noise Type"] = df["Noise Type"].str.capitalize()

model_names = sorted(df["Model"].unique())
cols = ["Dataset", "Noise Type"]
for model in model_names:
    cols.extend([f"{model} Min", f"{model} Mean", f"{model} Max"])

display_names = {
    "cifar10":   "CIFAR10",
    "cifar100":  "CIFAR100",
    "TinyImageNet": "TinyImageNet"
}


latex_rows = []

# Header
latex_rows.append("\\toprule")
latex_rows.append("Dataset & Noise Type & Min & Mean & Max  \\\\")
latex_rows.append("\\midrule")

for dataset in dataset_order:
    df_subset = df[(df["Dataset"] == dataset) & (df["Model"] == 1)]
    noise_types = ["Gaussian", "Laplace", "Uniform"]

    for i, noise in enumerate(noise_types):
        sub = df_subset[df_subset["Noise Type"] == noise]
        min_val = sub["N_min"].values[0]
        mean_val = sub["N_mean"].values[0]
        max_val = sub["N_max"].values[0]

        row = []

        if i == 0:
            # Add \multirow only once for the first row of the dataset block
            row.append(f"\\multirow{{3}}{{*}}{{{display_names[dataset]}}}")
        else:
            row.append("")  # Leave empty to align with multirow

        # Use ceil for mean value
        mean_val_rounded = math.ceil(mean_val)

        row.extend([noise, f"{min_val:.0f}", f"{mean_val_rounded}", f"{max_val:.0f}"])
        latex_rows.append(" & ".join(row) + " \\\\")
    
    latex_rows.append("\\midrule")


# Wrap in full LaTeX table
latex_wrapped = (
    "\\begin{table}[htbp]\n"
    "\\caption{Mean ± standard deviation of synthetic‐embedding counts across five independent runs (columns 1–5) for each noise distribution on CIFAR10, CIFAR100 and TinyImageNet.}\n"
    "\\centering\n"
    "\\begin{tabular}{lcccr}\n"
    + "\n".join(latex_rows) + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\label{tab:n_sample}\n"
    "\\end{table}"
)

# Save LaTeX
with open("results_diff_sampling/N_sample_latextable.tex", "w", encoding='utf-8') as f:
    f.write(latex_wrapped)

# Optional preview
print(latex_wrapped)
