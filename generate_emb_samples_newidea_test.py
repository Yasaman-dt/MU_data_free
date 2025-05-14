import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from create_embeddings_utils import get_model
from sklearn.manifold import TSNE

# ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
DIR = "/projets/Zdehghani/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"
dataset_name = "cifar10"
model_name = "resnet18"
num_classes = 10
n_model=2
if dataset_name.lower() in ["cifar10", "cifar100"]:
    dataset_name_upper = dataset_name.upper()
else:
    dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"
    

checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"  # Set your actual checkpoint path
model = get_model(model_name, dataset_name_upper, num_classes, checkpoint_path=checkpoint_path_model) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if isinstance(model.fc, nn.Sequential):
    for module in model.fc:
        if isinstance(module, nn.Linear):
            fc_layer = module
            break
else:
    fc_layer = model.fc

fc_layer = fc_layer.to(device)


# Function to compute coefficient matrix R from weights
def compute_coefficient_matrix(weights):
    weights = weights.reshape(weights.shape[0], -1)
    norm_weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    return np.dot(norm_weights, norm_weights.T)


# Function to generate feature space samples
def generate_feature_samples(n_samples, Sigma, mean_vector, device='cuda'):
    feature_samples = torch.distributions.MultivariateNormal(
        torch.from_numpy(mean_vector).float().to(device),
        torch.from_numpy(Sigma).float().to(device)
    ).sample((n_samples,))
    return feature_samples

# Function to generate soft targets from feature space samples
def generate_soft_targets(model, feature_samples, layer_index, temperature=1.0, device='cuda'):
    with torch.no_grad():
        if layer_index == -1:  # Directly using generated feature samples
            soft_targets = F.softmax(feature_samples / temperature, dim=1)
            predicted_labels = torch.argmax(soft_targets, dim=1)   # shape: (n_samples,)
    return soft_targets, predicted_labels


def accumulate_per_class_samples(fc_layer, Sigma, mean_vector, device, 
                                  n_classes, n_per_class, 
                                  batch_size=1000, threshold=0.97):
    class_counts = {i: 0 for i in range(n_classes)}
    class_features = {i: [] for i in range(n_classes)}
    class_soft_targets = {i: [] for i in range(n_classes)}
    
    layer_index = -1
    while any(class_counts[c] < n_per_class for c in range(n_classes)):
        # Step 1: Generate a batch of synthetic features
        feature_samples = generate_feature_samples(batch_size, Sigma, mean_vector, device)
        soft_targets, predicted_labels = generate_soft_targets(fc_layer, feature_samples, layer_index, device=device)

        confidence_scores = torch.max(soft_targets, dim=1).values
        high_conf_mask = confidence_scores > threshold

        high_conf_features = feature_samples[high_conf_mask]
        high_conf_targets = soft_targets[high_conf_mask]
        high_conf_labels = predicted_labels[high_conf_mask]

        # Step 2: Accumulate up to n_per_class per class
        for i in range(high_conf_features.shape[0]):
            class_name = int(high_conf_labels[i].item())
            if class_counts[class_name] < n_per_class:
                class_features[class_name].append(high_conf_features[i].unsqueeze(0))
                class_soft_targets[class_name].append(high_conf_targets[i].unsqueeze(0))
                class_counts[class_name] += 1

        print(f"Current class counts: {class_counts}")

    # Step 3: Stack all features and soft targets
    all_features = torch.cat([torch.cat(class_features[c], dim=0) for c in range(n_classes)], dim=0)
    all_targets = torch.cat([torch.cat(class_soft_targets[c], dim=0) for c in range(n_classes)], dim=0)
    all_labels = torch.cat([torch.full((n_per_class,), c, dtype=torch.long) for c in range(n_classes)], dim=0)

    return all_features, all_targets, all_labels


weights = fc_layer.weight.detach().cpu().numpy()
R = compute_coefficient_matrix(weights)
sigma_val = 50
sigma_values = np.ones(R.shape[0]) * sigma_val
D = np.diag(sigma_values)
Sigma = np.dot(D, np.dot(R, D))
mean_vector = np.zeros(R.shape[0])
#n_samples=1000
#feature_samples = generate_feature_samples(n_samples, Sigma, mean_vector, device)
#soft_targets, predicted_labels = generate_soft_targets(fc_layer, feature_samples, layer_index, device=device)

num_per_class = 100

synthetic_features, synthetic_soft_targets, synthetic_labels = accumulate_per_class_samples(
    fc_layer=fc_layer,
    Sigma=Sigma,
    mean_vector=mean_vector,
    device=device,
    n_classes=num_classes,
    n_per_class=num_per_class,  # total will be 1000
    batch_size=1000,
    threshold=0.97)


embedding_dim = 512
lr = 0.01
num_iterations = 3000
temperature = 1
n_samples = num_per_class * num_classes

# === Step 2: Initialize embeddings to be optimized ===
optimized_embeddings = torch.randn(n_samples, embedding_dim, requires_grad=True, device=device)

# === Step 3: Optimize embeddings to match soft targets ===
optimizer = torch.optim.Adam([optimized_embeddings], lr=lr)
loss_fn = nn.KLDivLoss(reduction='batchmean')

for step in range(num_iterations):
    optimizer.zero_grad()

    logits_pred = fc_layer(optimized_embeddings)
    probs = F.softmax(logits_pred / temperature, dim=1)

    synthetic_soft_targets = synthetic_soft_targets.to(device)

    loss = loss_fn(probs, synthetic_soft_targets)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")


optimized_embeddings_np = optimized_embeddings.detach().cpu().numpy()

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
optimized_embeddings_2d = tsne.fit_transform(optimized_embeddings_np)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(optimized_embeddings_2d[:, 0], optimized_embeddings_2d[:, 1], c=synthetic_labels, cmap="tab10", s=20)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE of Optimized Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()



train_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_train_m{n_model}.npz"
train_embeddings_data = np.load(train_path)
real_embeddings = torch.tensor(train_embeddings_data["embeddings"])
real_labels = torch.tensor(train_embeddings_data["labels"])


def select_n_per_class(embeddings, labels, num_per_class, num_classes):
    selected_embeddings = []
    selected_labels = []

    for num_class in range(num_classes):
        cls_indices = (labels == num_class).nonzero(as_tuple=True)[0]
        if len(cls_indices) >= num_per_class:
            chosen_indices = cls_indices[torch.randperm(len(cls_indices))[:num_per_class]]
            selected_embeddings.append(embeddings[chosen_indices])
            selected_labels.append(labels[chosen_indices])
        else:
            print(f"Warning: Not enough samples for class {num_class}. Found only {len(cls_indices)}")

    selected_embeddings = torch.cat(selected_embeddings, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)
    return selected_embeddings, selected_labels


real_embeddings_par, real_labels_par = select_n_per_class(real_embeddings, real_labels, num_per_class=num_per_class, num_classes=num_classes)
print(real_embeddings_par.shape)  
print(real_labels_par.shape)      


synthetic_embeddings = optimized_embeddings.detach().cpu()
synthetic_labels_new = synthetic_labels + num_classes 

# === Combine Real and Synthetic Embeddings ===
combined_embeddings = torch.cat([real_embeddings_par, synthetic_embeddings], dim=0)
combined_labels = torch.cat([real_labels_par, synthetic_labels_new.clone().detach()], dim=0)

# === Reduce to 2D using t-SNE ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
combined_2d = tsne.fit_transform(combined_embeddings.numpy())

plt.figure(figsize=(10, 7))
scatter = plt.scatter(combined_2d[:, 0], combined_2d[:, 1], c=combined_labels, cmap='tab20', s=10)
plt.colorbar(scatter, ticks=range(20), label='Class')
plt.title("t-SNE: Real (0–9) vs Synthetic (10–19) Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()

