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
    "cifar10": 10,
    #"cifar100": 100,
    #"TinyImageNet": 200,
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

    for n_model in model_indices:
        checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"
        model = get_model(model_name, dataset_name.upper(), num_classes, checkpoint_path=checkpoint_path_model)
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
    output_csv_path = os.path.join(DIR, f"all_class_probabilities.csv")
    
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        all_results_df = pd.concat([existing_df, all_results_df], ignore_index=True)
    
    all_results_df.to_csv(output_csv_path, index=False)
    print(f"Saved all results to: {output_csv_path}")





all_results_df = pd.read_csv("all_class_probabilities.csv")


# Filter for CIFAR10 only
subset = all_results_df[all_results_df["Dataset"] == "cifar10"]

plt.figure(figsize=(16, 7))
sns.barplot(
    data=subset,
    x="Class", y="Proportion", hue="Noise Type",
    errorbar="sd"
)
plt.title("Mean Proportion per Class with Std Dev for Each Noise Type (CIFAR10)")
plt.xlabel("Class")
plt.ylabel("Mean Proportion")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("barplot_class_distribution_cifar10.png")  # Optional: save the plot
plt.show()