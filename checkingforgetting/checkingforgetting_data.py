import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from create_embeddings_utils import get_model
from torch.utils.data import DataLoader, TensorDataset
import os

# ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
DIR = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"
dataset_name = "cifar10"
model_name = "resnet18"
num_classes = 10
forget_classes = list(range(num_classes))  # or a subset if needed

n_model=1
batch_size = 1024
method="RE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if dataset_name.lower() in ["cifar10", "cifar100"]:
    dataset_name_upper = dataset_name.upper()
else:
    dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"
    
if dataset_name in ["cifar10", "cifar100"]:
    dataset_name_lower = dataset_name.lower()
else:
    dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"

          
            
train_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_train_m{n_model}.npz"

if dataset_name_lower == "TinyImageNet":
    test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_val_m{n_model}.npz"
else:
    test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_test_m{n_model}.npz"
full_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_full_m{n_model}.npz"

train_embeddings_data = np.load(train_path)
test_embeddings_data = np.load(test_path)
full_embeddings_data = np.load(full_path)


# Access the embeddings and labels
train_emb = train_embeddings_data["embeddings"]  # The embeddings for the training data
train_labels = train_embeddings_data["labels"]   # The labels for the training data

test_emb = test_embeddings_data["embeddings"]  # The embeddings for the training data
test_labels = test_embeddings_data["labels"]   # The labels for the training data

full_emb = full_embeddings_data["embeddings"]  # The embeddings for the training data
full_labels = full_embeddings_data["labels"]   # The labels for the training data

train_emb = torch.tensor(train_emb, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

test_emb = torch.tensor(test_emb, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

full_emb = torch.tensor(full_emb, dtype=torch.float32)
full_labels = torch.tensor(full_labels, dtype=torch.long)

print("Train:", train_emb.shape, train_labels.shape)
print("Test:", test_emb.shape, test_labels.shape)
print("Full:", full_emb.shape, full_labels.shape)


for forget_class in forget_classes:
    print(f"Processing forget_class = {forget_class}")
    
    checkpoint_path_model = f"{DIR}/checkingforgetting/out_synth_gaussian/samples_per_class_5000/CR/{dataset_name}/{method}/models/unlearned_model_{method}_m{n_model}_seed_42_class_{forget_class}.pth"  # Set your actual checkpoint path
    model = get_model(model_name, dataset_name_upper, num_classes, checkpoint_path=checkpoint_path_model) 
  
    model.eval()
    fc_layer = model.fc.to(device)

    # Dataloaders using full data (not filtering forget class)
    train_loader = DataLoader(TensorDataset(train_emb, train_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_emb, test_labels), batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(TensorDataset(full_emb, full_labels), batch_size=batch_size, shuffle=False)

    def compute_logits(dataloader, fc_layer, device):
        all_logits = []
        all_probs = []
        all_labels = []
    
        with torch.no_grad():
            for emb, labels in dataloader:
                emb = emb.to(device)       # <-- Move embeddings to the same device as fc_layer
                logits = fc_layer(emb)
                probs = F.softmax(logits, dim=1)  # convert to probabilities

                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
    
        return torch.cat(all_logits).numpy(), torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()

    # Compute and save
    train_logits, train_probs, train_targets = compute_logits(train_loader, fc_layer, device)
    test_logits, test_probs, test_targets = compute_logits(test_loader, fc_layer, device)
    full_logits, full_probs, full_targets = compute_logits(full_loader, fc_layer, device)

    train_label = (train_targets == forget_class).astype(np.uint8)
    test_label = (test_targets == forget_class).astype(np.uint8)
    full_label = (full_targets == forget_class).astype(np.uint8)

    save_dir = f"{DIR}/checkingforgetting/{embeddings_folder}/{dataset_name_upper}/{method}/logits_forget_class_{forget_class}"
    os.makedirs(save_dir, exist_ok=True)

    np.savez(f"{save_dir}/train_logits_m{n_model}.npz",
             logits=train_logits,
             probs=train_probs,
             real_labels=train_targets,
             forget_labels=train_label)
    
    np.savez(f"{save_dir}/test_logits_m{n_model}.npz",
             logits=test_logits,
             probs=test_probs,
             real_labels=test_targets,
             forget_labels=test_label)
    
    np.savez(f"{save_dir}/full_logits_m{n_model}.npz",
             logits=full_logits,
             probs=full_probs,
             real_labels=full_targets,
             forget_labels=full_label)
