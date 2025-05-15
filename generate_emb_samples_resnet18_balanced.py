import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from create_embeddings_utils import get_model
from torch.utils.data import DataLoader, TensorDataset


def generate_emb_samples_balanced(B_matrix, num_classes, samples_per_class, sigma_range, resnet_model, device='cuda'):
    """
    Generates synthetic feature embeddings with a fixed number of samples per class.
    """
    weights = torch.tensor(B_matrix, dtype=torch.float32, device=device)

    def generate_feature_samples(n_samples, Sigma, mean_vector, device):
        return torch.distributions.MultivariateNormal(
            torch.from_numpy(mean_vector).float().to(device),
            torch.from_numpy(Sigma).float().to(device)
        ).sample((n_samples,))

    def compute_coefficient_matrix(weights):
        norm_weights = weights / torch.norm(weights, dim=1, keepdim=True)
        return torch.matmul(norm_weights, norm_weights.T)

    R = compute_coefficient_matrix(weights)

    def optimize_sigma(sigma_range):
        best_sigma, best_entropy, best_Sigma = sigma_range[0], float('inf'), None
        for sigma_val in sigma_range:
            sigma_values = torch.ones(R.shape[0], device=device) * sigma_val
            D = torch.diag(sigma_values)
            Sigma = torch.matmul(D, torch.matmul(R, D))

            mean_vector = np.zeros(R.shape[0])
            feature_samples = generate_feature_samples(100, Sigma.cpu().numpy(), mean_vector, device)
            entropy = -torch.sum(feature_samples * torch.log(feature_samples + 1e-10), dim=1).mean().item()

            if entropy < best_entropy or best_Sigma is None:
                best_entropy, best_sigma, best_Sigma = entropy, sigma_val, Sigma
            #print(best_sigma)
        if best_Sigma is None:
            best_Sigma = torch.eye(R.shape[0], device=device)

        return best_sigma, best_Sigma

    #best_sigma, best_Sigma = optimize_sigma(sigma_range)
    #print(best_sigma)

    fixed_sigma = 5
    sigma_values = torch.ones(R.shape[0], device=device) * fixed_sigma
    D = torch.diag(sigma_values)
    best_Sigma = torch.matmul(D, torch.matmul(R, D))

    output_size = 6
    expected_feature_size = 512 * 6 * 6
    mean_vector = np.zeros(best_Sigma.shape[0])
    total_needed = samples_per_class * num_classes

    collected_features = []
    collected_labels = []

    resnet_model.to(device)
    resnet_model.eval()

    with torch.no_grad():
        for cls in range(num_classes):
            class_features = []
            attempts = 0
            while len(class_features) < samples_per_class:
                # Over-generate to filter later
                if samples_per_class >= 301:
                    batch_size = int(samples_per_class * min(4, 18 * num_classes / samples_per_class))
                else:
                    batch_size = int(samples_per_class * max(4, 18 * num_classes / samples_per_class))
                #feature_samples = generate_feature_samples(batch_size, best_Sigma.cpu().numpy(), mean_vector, device)

                # if best_Sigma.shape[0] == expected_feature_size:
                #     feature_samples = feature_samples.view(batch_size, 512, output_size, output_size)
                #     feature_samples = F.adaptive_avg_pool2d(feature_samples, (1, 1))
                #     feature_samples = feature_samples.view(batch_size, -1)

                feature_samples = torch.randn(batch_size, best_Sigma.shape[0], device=device)

                logits = resnet_model.fc(feature_samples)
                probs = F.softmax(logits, dim=1)
                predicted = torch.argmax(probs, dim=1)

                # Select those predicted as the current class
                mask = predicted == cls
                selected = feature_samples[mask]
                class_features.append(selected[:samples_per_class - len(class_features)])
                attempts += 1

                if attempts > 1000:  # Fail-safe
                    print(f"Warning: Class {cls} took too many attempts to fill quota.")
                    break


            class_features = torch.cat(class_features, dim=0)
            class_features = class_features[:samples_per_class]  # Just in case over-collected
            class_labels = torch.full((class_features.size(0),), cls, dtype=torch.long, device=device)
            collected_features.append(class_features)
            collected_labels.append(class_labels)




    sample_features = torch.cat(collected_features, dim=0)
    sample_labels = torch.cat(collected_labels, dim=0)

    # Recalculate final probability array
    with torch.no_grad():
        probability_array = F.softmax(resnet_model.fc(sample_features), dim=1).cpu().numpy()

    return sample_features, sample_labels, probability_array


def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during inference
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy        
        
        
# # ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
# DIR = "/projets/Zdehghani/MU_data_free"
# checkpoint_folder = "checkpoints"
# weights_folder = "weights"
# embeddings_folder = "embeddings"
# dataset_name = "cifar10"
# model_name = "resnet18"
# num_classes = 10
# n_model=3
# if dataset_name.lower() in ["cifar10", "cifar100"]:
#     dataset_name_upper = dataset_name.upper()
# else:
#     dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"
    
# if dataset_name in ["cifar10", "cifar100"]:
#     dataset_name_lower = dataset_name.lower()
# else:
#     dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"

# checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"  # Set your actual checkpoint path
# model = get_model(model_name, dataset_name_upper, num_classes, checkpoint_path=checkpoint_path_model) 
            
# # Load the saved matrix_B_224
# matrix_B_224 = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/matrix_B_224_m{n_model}.npy"
# B_numpy = np.load(matrix_B_224)  # Shape: (512, 2304)
  
# # Parameters
# samples_per_class = 100  # Generate 1000 samples per class
# sigma_range = np.linspace(0.5, 6, 3)  # Range for sigma optimization
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # Run the function
# train_emb_synth, train_labels_synth, probability_array = generate_emb_samples_balanced(B_numpy, num_classes, samples_per_class, sigma_range, model, device=device)
# # Print summary statistics
# print(f"Generated feature tensor shape: {train_emb_synth.shape}")  # Expected: (10000, 512)
# print(f"Generated label tensor shape: {train_labels_synth.shape}")  # Expected: (10000,)
# print(f"Probability array shape: {probability_array.shape}")  # Expected: (10000, 10)
# print(f"Unique classes in generated labels: {torch.unique(train_labels_synth)}")
# # Validate class distribution
# unique, counts = torch.unique(train_labels_synth, return_counts=True)
# class_distribution = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
# print(f"Class distribution: {class_distribution}")
# # Validate feature statistics
# mean_features = train_emb_synth.mean(dim=0)
# std_features = train_emb_synth.std(dim=0)
# print(f"Feature mean: {mean_features.mean().item():.4f}, Feature std: {std_features.mean().item():.4f}")
# # Check for NaNs or invalid values
# if torch.isnan(train_emb_synth).any():
#     print("Warning: NaN values detected in generated features.")
# if torch.isnan(train_labels_synth).any():
#     print("Warning: NaN values detected in generated labels.")
# print("Feature generation test completed successfully!")
# # ------------------ Load Real Embeddings ------------------
# train_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_train_m{n_model}.npz"
# test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_test_m{n_model}.npz"
# train_embeddings_data = np.load(train_path)
# test_embeddings_data = np.load(test_path)
# train_emb_real = torch.tensor(train_embeddings_data["embeddings"], dtype=torch.float32)
# train_labels_real = torch.tensor(train_embeddings_data["labels"], dtype=torch.long)
# test_emb_real = torch.tensor(test_embeddings_data["embeddings"], dtype=torch.float32)
# test_labels_real = torch.tensor(test_embeddings_data["labels"], dtype=torch.long)
# # Move data to device
# train_emb_real, train_labels_real = train_emb_real.to(device), train_labels_real.to(device)
# test_emb_real, test_labels_real= test_emb_real.to(device), test_labels_real.to(device)
# batch_size = 256
# # Create DataLoaders for Real and Synthetic Data
# train_loader_real = DataLoader(TensorDataset(train_emb_real, train_labels_real), batch_size=batch_size, shuffle=True)
# test_loader_real = DataLoader(TensorDataset(test_emb_real, test_labels_real), batch_size=batch_size, shuffle=False)
# synth_train_loader = DataLoader(TensorDataset(train_emb_synth, train_labels_synth), batch_size=batch_size, shuffle=True)
# # ------------------ Evaluate Model Accuracy ------------------
# fc_layer = model.fc.to(device)
# # Evaluate accuracy using synthetic embeddings
# synth_train_accuracy = evaluate_model(fc_layer, synth_train_loader, device)
# print(f"Synthetic Train Accuracy: {synth_train_accuracy:.2f}%")

# unique, counts = torch.unique(train_labels_synth, return_counts=True)
# print("Synthetic label distribution:", dict(zip(unique.tolist(), counts.tolist())))
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt



# # Run the function
# real_features = train_emb_real
# synth_features = train_emb_synth
# real_labels = train_labels_real 
# synth_labels = train_labels_synth 
# n_samples=1000
# seed=42
# torch.manual_seed(seed)
# np.random.seed(seed)

# n_classes = real_labels.max().item() + 1

# def subsample_by_class(features, labels, n_per_class):
#     selected_features, selected_labels = [], []
#     for cls in range(n_classes):
#         idxs = (labels == cls).nonzero(as_tuple=True)[0]
#         selected = idxs[torch.randperm(len(idxs))[:n_per_class]]
#         selected_features.append(features[selected])
#         selected_labels.append(labels[selected])
#     return torch.cat(selected_features), torch.cat(selected_labels)

# # Subsample
# real_sub, real_lbls = subsample_by_class(real_features, real_labels, n_samples // n_classes)
# synth_sub, synth_lbls = subsample_by_class(synth_features, synth_labels, n_samples // n_classes)

# # Compute t-SNE separately
# tsne_real = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=seed)
# tsne_synth = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=10000, random_state=seed)
# synth_sub = (synth_sub - synth_sub.min())/synth_sub.max()

# tsne_real_result = tsne_real.fit_transform(real_sub.cpu().numpy())
# tsne_synth_result = tsne_synth.fit_transform(synth_sub.cpu().numpy())

# # Plot: Real
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# for cls in range(n_classes):
#     idxs = real_lbls.cpu().numpy() == cls
#     plt.scatter(tsne_real_result[idxs, 0], tsne_real_result[idxs, 1], label=f"Class {cls}", alpha=0.6, s=20)
# plt.title("t-SNE of Real Embeddings")
# plt.legend()
# plt.grid(True)

# # Plot: Synthetic
# plt.subplot(1, 2, 2)
# for cls in range(n_classes):
#     idxs = synth_lbls.cpu().numpy() == cls
#     plt.scatter(tsne_synth_result[idxs, 0], tsne_synth_result[idxs, 1], label=f"Class {cls}", alpha=0.6, s=20)
# plt.title("t-SNE of Synthetic Embeddings")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("tsne_real_vs_synthetic_separate.png", dpi=300)  # <- Add this line

# plt.show()
