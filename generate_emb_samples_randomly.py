import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from create_embeddings_utils import get_model
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn



def _last_linear(module: nn.Module) -> nn.Linear:
    """
    Find the last nn.Linear inside a module (e.g., inside an nn.Sequential).
    Raises if none is found.
    """
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            return m
    raise AttributeError("No nn.Linear found inside the given module.")

def _get_classifier_and_dim(net):
    """
    Returns (classifier_module, final_linear, in_features, out_features)
    - Works for ViT (.heads / .head / .classifier), ResNet (.fc), Swin (.head), etc.
    - final_linear is the last nn.Linear inside the classifier stack.
    """
    # Try common classifier attribute names in priority order
    for attr in ("heads", "head", "classifier", "fc", "classif"):
        if hasattr(net, attr):
            clf = getattr(net, attr)
            if isinstance(clf, nn.Linear):
                # classifier *is* the final linear
                return clf, clf, clf.in_features, clf.out_features
            else:
                last = _last_linear(clf)  # final Linear inside the head
                return clf, last, last.in_features, last.out_features

    # Some timm models route through get_classifier()
    if hasattr(net, "get_classifier"):
        clf = net.get_classifier()
        if isinstance(clf, nn.Linear):
            return clf, clf, clf.in_features, clf.out_features
        else:
            last = _last_linear(clf)
            return clf, last, last.in_features, last.out_features

    raise AttributeError("Could not locate a classifier layer (heads/head/classifier/fc).")
    
    
    
def generate_emb_samples_balanced(num_classes, samples_per_class, net, noise_type, device='cuda'):
    batch_size = 2000

    net.eval().to(device)
    clf, final_linear, embedding_dim, out_dim = _get_classifier_and_dim(net)
    final_linear = final_linear.to(device).eval()  # <- use this for [N, C] embeddings

    all_sample_probs = []
    class_counts = {i: 0 for i in range(num_classes)}
    class_features = {i: [] for i in range(num_classes)}
    class_soft_targets = {i: [] for i in range(num_classes)}

    while any(class_counts[c] < samples_per_class for c in range(num_classes)):
        # Generate random synthetic samples in *feature space*
        if noise_type == "gaussian":
            feature_samples = torch.randn(batch_size, embedding_dim, device=device)
        elif noise_type == "bernoulli":
            feature_samples = torch.bernoulli(torch.full((batch_size, embedding_dim), 0.5, device=device))
        elif noise_type == "uniform":
            feature_samples = torch.empty(batch_size, embedding_dim, device=device).uniform_(-1, 1)
        elif noise_type == "laplace":
            feature_samples = torch.distributions.Laplace(0.0, 1.0).sample((batch_size, embedding_dim)).to(device)
        elif noise_type == "gumbel":
            feature_samples = torch.distributions.Gumbel(0.0, 1.0).sample((batch_size, embedding_dim)).to(device)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        # “probabilities” tracker (log-density under N(0, I))
        norm_squared = feature_samples.pow(2).sum(dim=1)
        log_prob = -0.5 * (embedding_dim * np.log(2 * np.pi) + norm_squared.detach().cpu().numpy())
        probabilities = log_prob

        with torch.no_grad():
            logits = final_linear(feature_samples)   # works for both Sequential(heads) and Linear(fc)
            soft_targets = F.softmax(logits, dim=1)
            predicted_labels = torch.argmax(soft_targets, dim=1)

        for i in range(batch_size):
            class_name = int(predicted_labels[i])
            if class_counts[class_name] < samples_per_class:
                class_features[class_name].append(feature_samples[i].unsqueeze(0))
                class_soft_targets[class_name].append(soft_targets[i].unsqueeze(0))
                class_counts[class_name] += 1
                all_sample_probs.append(probabilities[i])

        print(f"Current class counts: {class_counts}")

    # Stitch per-class samples
    all_features, all_labels, all_soft_targets = [], [], []
    for class_name in range(num_classes):
        feats = torch.cat(class_features[class_name], dim=0)
        targets = torch.cat(class_soft_targets[class_name], dim=0)
        labels = torch.full((feats.size(0),), class_name, dtype=torch.long, device=device)
        all_features.append(feats)
        all_labels.append(labels)
        all_soft_targets.append(targets)

    sample_features = torch.cat(all_features, dim=0)
    sample_labels = torch.cat(all_labels, dim=0)
    probability_array = torch.cat(all_soft_targets, dim=0).cpu().numpy()
    return sample_features, sample_labels, probability_array, all_sample_probs


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
# n_model=1
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
            
# # # Load the saved matrix_B_224
# # matrix_B_224 = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/matrix_B_224_m{n_model}.npy"
# # B_numpy = np.load(matrix_B_224)  # Shape: (512, 2304)
  
# # # Parameters
# # samples_per_class = 100  # Generate 1000 samples per class
# # sigma_range = np.linspace(0.5, 6, 3)  # Range for sigma optimization
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # Run the function
# # train_emb_synth, train_labels_synth, probability_array = generate_emb_samples_balanced(B_numpy, num_classes, samples_per_class, sigma_range, model, device=device)
# # # Print summary statistics
# # print(f"Generated feature tensor shape: {train_emb_synth.shape}")  # Expected: (10000, 512)
# # print(f"Generated label tensor shape: {train_labels_synth.shape}")  # Expected: (10000,)
# # print(f"Probability array shape: {probability_array.shape}")  # Expected: (10000, 10)
# # print(f"Unique classes in generated labels: {torch.unique(train_labels_synth)}")
# # # Validate class distribution
# # unique, counts = torch.unique(train_labels_synth, return_counts=True)
# # class_distribution = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
# # print(f"Class distribution: {class_distribution}")
# # # Validate feature statistics
# # mean_features = train_emb_synth.mean(dim=0)
# # std_features = train_emb_synth.std(dim=0)
# # print(f"Feature mean: {mean_features.mean().item():.4f}, Feature std: {std_features.mean().item():.4f}")
# # # Check for NaNs or invalid values
# # if torch.isnan(train_emb_synth).any():
# #     print("Warning: NaN values detected in generated features.")
# # if torch.isnan(train_labels_synth).any():
# #     print("Warning: NaN values detected in generated labels.")
# # print("Feature generation test completed successfully!")
# # # ------------------ Load Real Embeddings ------------------
# # train_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_train_m{n_model}.npz"
# # test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_test_m{n_model}.npz"
# # train_embeddings_data = np.load(train_path)
# # test_embeddings_data = np.load(test_path)
# # train_emb_real = torch.tensor(train_embeddings_data["embeddings"], dtype=torch.float32)
# # train_labels_real = torch.tensor(train_embeddings_data["labels"], dtype=torch.long)
# # test_emb_real = torch.tensor(test_embeddings_data["embeddings"], dtype=torch.float32)
# # test_labels_real = torch.tensor(test_embeddings_data["labels"], dtype=torch.long)
# # # Move data to device
# # train_emb_real, train_labels_real = train_emb_real.to(device), train_labels_real.to(device)
# # test_emb_real, test_labels_real= test_emb_real.to(device), test_labels_real.to(device)
# # batch_size = 256
# # # Create DataLoaders for Real and Synthetic Data
# # train_loader_real = DataLoader(TensorDataset(train_emb_real, train_labels_real), batch_size=batch_size, shuffle=True)
# # test_loader_real = DataLoader(TensorDataset(test_emb_real, test_labels_real), batch_size=batch_size, shuffle=False)
# # synth_train_loader = DataLoader(TensorDataset(train_emb_synth, train_labels_synth), batch_size=batch_size, shuffle=True)
# # # ------------------ Evaluate Model Accuracy ------------------
# # fc_layer = model.fc.to(device)
# # # Evaluate accuracy using synthetic embeddings
# # synth_train_accuracy = evaluate_model(fc_layer, synth_train_loader, device)
# # print(f"Synthetic Train Accuracy: {synth_train_accuracy:.2f}%")

# # unique, counts = torch.unique(train_labels_synth, return_counts=True)
# # print("Synthetic label distribution:", dict(zip(unique.tolist(), counts.tolist())))
# # from sklearn.manifold import TSNE
# # import matplotlib.pyplot as plt



# # # Run the function
# # real_features = train_emb_real
# # synth_features = train_emb_synth
# # real_labels = train_labels_real 
# # synth_labels = train_labels_synth 
# # n_samples=1000
# # seed=42
# # torch.manual_seed(seed)
# # np.random.seed(seed)

# # n_classes = real_labels.max().item() + 1

# # def subsample_by_class(features, labels, n_per_class):
# #     selected_features, selected_labels = [], []
# #     for cls in range(n_classes):
# #         idxs = (labels == cls).nonzero(as_tuple=True)[0]
# #         selected = idxs[torch.randperm(len(idxs))[:n_per_class]]
# #         selected_features.append(features[selected])
# #         selected_labels.append(labels[selected])
# #     return torch.cat(selected_features), torch.cat(selected_labels)

# # # Subsample
# # real_sub, real_lbls = subsample_by_class(real_features, real_labels, n_samples // n_classes)
# # synth_sub, synth_lbls = subsample_by_class(synth_features, synth_labels, n_samples // n_classes)

# # # Compute t-SNE separately
# # tsne_real = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=seed)
# # tsne_synth = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=10000, random_state=seed)
# # synth_sub = (synth_sub - synth_sub.min())/synth_sub.max()

# # tsne_real_result = tsne_real.fit_transform(real_sub.cpu().numpy())
# # tsne_synth_result = tsne_synth.fit_transform(synth_sub.cpu().numpy())

# # # Plot: Real
# # plt.figure(figsize=(10, 5))
# # plt.subplot(1, 2, 1)
# # for cls in range(n_classes):
# #     idxs = real_lbls.cpu().numpy() == cls
# #     plt.scatter(tsne_real_result[idxs, 0], tsne_real_result[idxs, 1], label=f"Class {cls}", alpha=0.6, s=20)
# # plt.title("t-SNE of Real Embeddings")
# # plt.legend()
# # plt.grid(True)

# # # Plot: Synthetic
# # plt.subplot(1, 2, 2)
# # for cls in range(n_classes):
# #     idxs = synth_lbls.cpu().numpy() == cls
# #     plt.scatter(tsne_synth_result[idxs, 0], tsne_synth_result[idxs, 1], label=f"Class {cls}", alpha=0.6, s=20)
# # plt.title("t-SNE of Synthetic Embeddings")
# # plt.legend()
# # plt.grid(True)

# # plt.tight_layout()
# # plt.savefig("tsne_real_vs_synthetic_separate.png", dpi=300)  # <- Add this line

# # plt.show()
# import torch
# import matplotlib.pyplot as plt
# import numpy as np


# model.eval()
# fc_layer = model.fc.to(device)

# # Define embedding_dim
# bbone = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
# embedding_dim = bbone(torch.randn(1, 3, 64, 64).to(device)).shape[1]

# # ------------------ Noise Sampling ------------------
# def sample_noise(noise_type, size, device='cuda'):
#     if noise_type == "gaussian":
#         return torch.randn(size, device=device)
#     elif noise_type == "bernoulli":
#         return torch.bernoulli(torch.full(size, 0.5, device=device))
#     elif noise_type == "uniform":
#         return torch.empty(size, device=device).uniform_(-1, 1)
#     elif noise_type == "laplace":
#         return torch.distributions.Laplace(0.0, 1.0).sample(size).to(device)
#     elif noise_type == "gumbel":
#         return torch.distributions.Gumbel(0.0, 1.0).sample(size).to(device)
#     else:
#         raise ValueError(f"Unsupported noise type: {noise_type}")


# # ------------------ Analysis Function ------------------
# def analyze_class_probabilities_from_noise(fc_layer, embedding_dim, device, num_samples, noise_type):
#     size = (num_samples, embedding_dim)
#     feature_samples = sample_noise(noise_type, size, device)
#     print(f"\n=== Class prediction distribution over {noise_type} noise ===")

#     with torch.no_grad():
#         logits = fc_layer(feature_samples)
#         preds = torch.argmax(logits, dim=1)

#     class_counts = torch.bincount(preds, minlength=fc_layer.out_features).cpu().numpy()
#     class_probs = class_counts / num_samples

#     for i, (count, prob) in enumerate(zip(class_counts, class_probs)):
#         print(f"Class {i}: count={count}, proportion={prob:.4f}")

#     return class_probs

# # ------------------ Run for All Noise Types ------------------
# noise_types = ["gaussian", "bernoulli", "uniform", "laplace", "gumbel"]
# all_class_probs = {}

# for noise in noise_types:
#     class_probs = analyze_class_probabilities_from_noise(fc_layer, embedding_dim, device, 1000000, noise)
#     all_class_probs[noise] = class_probs

# # ------------------ Plotting ------------------
# def plot_all_distributions(all_class_probs):
#     num_classes = len(next(iter(all_class_probs.values())))
#     x = np.arange(num_classes)
#     width = 0.15

#     plt.figure(figsize=(14, 6))
#     for i, (noise_type, probs) in enumerate(all_class_probs.items()):
#         offset = (i - len(all_class_probs)/2) * width + width/2
#         plt.bar(x + offset, probs, width, label=noise_type.capitalize())

#     plt.xlabel("Class")
#     plt.ylabel("Proportion of Predictions")
#     plt.title("Class Prediction Distribution Across Different Noise Types")
#     plt.xticks(x)
#     plt.ylim(top=0.210)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"class_distribution_all_noises_{dataset_name}_{n_model}.png")
#     plt.show()

# plot_all_distributions(all_class_probs)