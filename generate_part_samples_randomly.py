import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from create_embeddings_utils import get_model
from torch.utils.data import DataLoader, TensorDataset


class TruncatedResNet(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3_0 = original_model.layer3[0]
        self.layer3_1_conv1 = original_model.layer3[1].conv1
        self.layer3_1_bn1 = original_model.layer3[1].bn1
        self.layer3_1_relu = original_model.layer3[1].relu
        self.layer3_1_conv2 = original_model.layer3[1].conv2
        self.layer3_1_bn2 = original_model.layer3[1].bn2
        self.layer4_0 = original_model.layer4[0]
        self.layer4_1_conv1 = original_model.layer4[1].conv1
        self.layer4_1_bn1 = original_model.layer4[1].bn1
        self.layer4_1_relu = original_model.layer4[1].relu
        # self.layer4_1_conv2 = original_model.layer4[1].conv2
        # self.layer4_1_bn2 = original_model.layer4[1].bn2
        # self.avgpool = original_model.avgpool
        # self.flatten = torch.nn.Flatten()
        #self.fc = original_model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3_0(x)
        x = self.layer3_1_conv1(x)
        x = self.layer3_1_bn1(x)
        x = self.layer3_1_relu(x)
        x = self.layer3_1_conv2(x)
        x = self.layer3_1_bn2(x)
        x = self.layer4_0(x)
        x = self.layer4_1_conv1(x)
        x = self.layer4_1_bn1(x)
        x = self.layer4_1_relu(x)
        # x = self.layer4_1_conv2(x)
        # x = self.layer4_1_bn2(x)
        # x = self.avgpool(x)
        # x = self.flatten(x)
        #x = self.fc(x)
        return x



class RemainingResNet(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # self.conv1 = original_model.conv1
        # self.bn1 = original_model.bn1
        # self.relu = original_model.relu
        # self.maxpool = original_model.maxpool
        # self.layer1 = original_model.layer1
        # self.layer2 = original_model.layer2
        # self.layer3_0 = original_model.layer3[0]
        # self.layer3_1_conv1 = original_model.layer3[1].conv1
        # self.layer3_1_bn1 = original_model.layer3[1].bn1
        # self.layer3_1_relu = original_model.layer3[1].relu
        # self.layer3_1_conv2 = original_model.layer3[1].conv2
        # self.layer3_1_bn2 = original_model.layer3[1].bn2
        # self.layer4_0 = original_model.layer4[0]
        # self.layer4_1_conv1 = original_model.layer4[1].conv1
        # self.layer4_1_bn1 = original_model.layer4[1].bn1
        # self.layer4_1_relu = original_model.layer4[1].relu
        self.layer4_1_conv2 = original_model.layer4[1].conv2
        self.layer4_1_bn2 = original_model.layer4[1].bn2
        self.avgpool = original_model.avgpool
        self.flatten = torch.nn.Flatten()
        self.fc = original_model.fc

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3_0(x)
        # x = self.layer3_1_conv1(x)
        # x = self.layer3_1_bn1(x)
        # x = self.layer3_1_relu(x)
        # x = self.layer3_1_conv2(x)
        # x = self.layer3_1_bn2(x)
        # x = self.layer4_0(x)
        # x = self.layer4_1_conv1(x)
        # x = self.layer4_1_bn1(x)
        # x = self.layer4_1_relu(x)
        x = self.layer4_1_conv2(x)
        x = self.layer4_1_bn2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



def generate_emb_samples_balanced(num_classes, samples_per_class, resnet_model, dataset, device='cuda'):
    
    batch_size = 2000
    Truncatedmodel = TruncatedResNet(resnet_model).to(device)
    Remainingmodel = RemainingResNet(resnet_model).to(device)
    if dataset == "TinyImageNet":
        a = torch.randn(1, 3, 64, 64).to(device)
    elif dataset in ["cifar10", "cifar100"]:
        a = torch.randn(1, 3, 32, 32).to(device)

    embedding = Truncatedmodel(a)
    print(embedding.shape)
    embedding_shape = embedding.shape[1:]
    embedding_dim = int(np.prod(embedding_shape))
    print(embedding_dim)
    Truncatedmodel.eval()
    Remainingmodel.eval()

    
  
    class_counts = {i: 0 for i in range(num_classes)}
    class_features = {i: [] for i in range(num_classes)}
    class_soft_targets = {i: [] for i in range(num_classes)}



    while any(class_counts[c] < samples_per_class for c in range(num_classes)):
        # Generate random synthetic samples
        feature_samples = torch.randn(batch_size, *embedding_shape, device=device)
        #feature_samples = torch.randn(batch_size, embedding_dim, device=device)
        #feature_samples = feature_samples.view(batch_size, 512, 2, 2)
        #print(feature_samples.shape)
        with torch.no_grad():
            logits = Remainingmodel(feature_samples)
            soft_targets = F.softmax(logits, dim=1)
            predicted_labels = torch.argmax(soft_targets, dim=1)

        for i in range(batch_size):
            class_name = int(predicted_labels[i].item())
            if class_counts[class_name] < samples_per_class:
                class_features[class_name].append(feature_samples[i].unsqueeze(0).cpu())
                class_soft_targets[class_name].append(soft_targets[i].unsqueeze(0).cpu())
                class_counts[class_name] += 1

        del feature_samples, logits, soft_targets, predicted_labels
        torch.cuda.empty_cache()
        print(f"Current class counts: {class_counts}")

    # Combine all samples
    all_features = []
    all_labels = []
    all_soft_targets = []

    for class_name in range(num_classes):
        feats = torch.cat(class_features[class_name], dim=0)
        targets = torch.cat(class_soft_targets[class_name], dim=0)
        labels = torch.full((feats.size(0),), class_name, dtype=torch.long, device=device)

        all_features.append(feats)
        all_labels.append(labels)
        all_soft_targets.append(targets)

        del feats, targets, labels
        torch.cuda.empty_cache()
    
    sample_features = torch.cat(all_features, dim=0)
    sample_labels = torch.cat(all_labels, dim=0)
    probability_array = torch.cat(all_soft_targets, dim=0).cpu().numpy()

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
