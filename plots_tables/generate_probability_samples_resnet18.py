import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from create_embeddings_utils import get_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
DIR = "/projets/Zdehghani/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"
dataset_name = "cifar10"
model_name = "resnet18"
num_classes = 10
n_model=3
if dataset_name.lower() in ["cifar10", "cifar100"]:
   dataset_name_upper = dataset_name.upper()
else:
   dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"
   
if dataset_name in ["cifar10", "cifar100"]:
   dataset_name_lower = dataset_name.lower()
else:
   dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"

checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"  # Set your actual checkpoint path
model = get_model(model_name, dataset_name_upper, num_classes, checkpoint_path=checkpoint_path_model) 
           



# Function to compute coefficient matrix R from weights
def compute_coefficient_matrix(weights):
    weights = weights.reshape(weights.shape[0], -1)
    norm_weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    return np.dot(norm_weights, norm_weights.T)

# Selecting the feature space layer
layer_index = -1
if layer_index == -1:
    weights = model.fc.weight.detach().cpu().numpy()

R = compute_coefficient_matrix(weights)

# Function to generate feature space samples
def generate_feature_samples(n_samples, Sigma, mean_vector, device='cuda'):
    feature_samples = torch.distributions.MultivariateNormal(
        torch.from_numpy(mean_vector).float().to(device),
        torch.from_numpy(Sigma).float().to(device)
    ).sample((n_samples,))

    # shape = torch.tensor(mean_vector).shape
    # feature_samples = 5*torch.rand((n_samples, *shape))
    # print("mm", feature_samples.shape)
    return feature_samples

# Function to generate soft targets from feature space samples
def generate_soft_targets(teacher, feature_samples, layer_index, temperature=1.0, device='cuda'):
    with torch.no_grad():
        if layer_index == -1:  # Directly using generated feature samples
            soft_targets = F.softmax(feature_samples / temperature, dim=1)
    return soft_targets

# Function to optimize sigma (variance scaling)
def optimize_sigma(teacher, n_samples, layer_index, sigma_range, device='cuda'):
    best_sigma = None
    best_entropy = float('inf')
    best_Sigma = None

    for sigma_val in sigma_range:
        sigma_values = np.ones(R.shape[0]) * sigma_val
        D = np.diag(sigma_values)
        Sigma = np.dot(D, np.dot(R, D))


        mean_vector = np.zeros(R.shape[0])  # Ensure correct feature space dimension
        print(mean_vector.shape)

        feature_samples = generate_feature_samples(n_samples, Sigma, mean_vector, device)
        soft_targets = generate_soft_targets(teacher, feature_samples, layer_index, device=device)

        entropy = -torch.sum(soft_targets * torch.log(soft_targets + 1e-10), dim=1).mean().item()

        print(f"Sigma: {sigma_val}, Entropy: {entropy}")

        if entropy < best_entropy:
            best_entropy = entropy
            best_sigma = sigma_val
            best_Sigma = Sigma

    print(f"Best Sigma: {best_sigma} with Entropy: {best_entropy}")
    return best_sigma, best_Sigma

# Define sigma range and optimize
sigma_range = np.linspace(0.5, 5, num=20)  # 20 values from 0.5 to 10
best_sigma, best_Sigma = optimize_sigma(model, n_samples=100, layer_index=layer_index, sigma_range=sigma_range)


# Generate 10,000 feature samples using the best Sigma
num_samples = 1000
mean_vector = np.zeros(best_Sigma.shape[0])  # Ensuring correct feature space dimension

feature_samples = generate_feature_samples(num_samples, best_Sigma, mean_vector, device)

# Get soft targets (probabilities) from the teacher model
soft_targets = generate_soft_targets(model, feature_samples, layer_index=layer_index, device=device)
max_probs, pseudo_labels = torch.max(soft_targets, dim=1)  # shapes: (N,), (N,)

conf_mask = max_probs > 0.97


filtered_features = feature_samples[conf_mask]
filtered_labels = pseudo_labels[conf_mask]
filtered_probs = soft_targets[conf_mask]            # (M, C)

# Convert to NumPy arrays
filtered_feature_np = filtered_features.cpu().numpy()
filtered_label_np = filtered_labels.cpu().numpy()
filtered_probs_np = filtered_probs.cpu().numpy()  # Save full probability distribution

num_classes=10


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Step 3: Apply t-SNE on filtered samples
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(filtered_feature_np)

# Step 4: Plot t-SNE
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=filtered_label_np, cmap='tab10', s=40)
plt.title("t-SNE of High-Confidence Classifier Outputs (Prob > 0.97)")
plt.colorbar(scatter, ticks=range(num_classes), label='Class')
plt.grid(True)
plt.tight_layout()
plt.show()


tsne_result = tsne.fit_transform(filtered_probs_np)
# Step 4: Plot t-SNE
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=filtered_label_np, cmap='tab10', s=40)
plt.title("t-SNE of High-Confidence Classifier Outputs (Prob > 0.97)")
plt.colorbar(scatter, ticks=range(num_classes), label='Class')
plt.grid(True)
plt.tight_layout()
plt.show()


