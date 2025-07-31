import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from create_embeddings_utils import get_model
from torch.utils.data import DataLoader, TensorDataset
import os
from SVC_MIA import SVC_MIA
import pandas as pd
import os
import argparse
from MIA2 import membership_inference_attack 

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser(description="Run SVC MIA pipeline with model and method options.")
parser.add_argument('--n_model', type=int, default=1, help='Model index (e.g., 1 to 5)')
parser.add_argument('--method', type=str, default='original', help='Unlearning method to evaluate (e.g., original, retrained, FT, etc.)')
parser.add_argument('--model_name', type=str, default='resnet18', help='Model name (e.g., resnet18, vit, etc.)')
parser.add_argument('--source', type=str, default='real', choices=['real', 'synth'], help='Data source: real or synth')

args = parser.parse_args()

n_model = args.n_model
method = args.method
model_name = args.model_name
source = args.source


# ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
DIR = "/projets/Zdehghani/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"
dataset_name = "cifar10"
#model_name = "resnet18"
num_classes = 10
forget_classes = list(range(num_classes))  # or a subset if needed
#source = "real" #synth
#n_model=1
batch_size = 1024
#method="original"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed=42

if dataset_name.lower() in ["cifar10", "cifar100"]:
    dataset_name_upper = dataset_name.upper()
else:
    dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"
    
if dataset_name in ["cifar10", "cifar100"]:
    dataset_name_lower = dataset_name.lower()
else:
    dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"

          
#for n_model in range(1, 6):
print(f"Processing n_model = {n_model}")            
train_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_train_m{n_model}.npz"

if dataset_name_lower == "TinyImageNet":
    test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_val_m{n_model}.npz"
else:
    test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_test_m{n_model}.npz"
full_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_full_m{n_model}.npz"

train_embeddings_data = np.load(train_path)
test_embeddings_data = np.load(test_path)
full_embeddings_data = np.load(full_path)


train_data = np.load(train_path)
test_data = np.load(test_path)
full_data = np.load(full_path)

train_emb = torch.tensor(train_data["embeddings"], dtype=torch.float32)
train_labels = torch.tensor(train_data["labels"], dtype=torch.long)
test_emb = torch.tensor(test_data["embeddings"], dtype=torch.float32)
test_labels = torch.tensor(test_data["labels"], dtype=torch.long)
full_emb = torch.tensor(full_data["embeddings"], dtype=torch.float32)
full_labels = torch.tensor(full_data["labels"], dtype=torch.long)

print("Train:", train_emb.shape, train_labels.shape)
print("Test:", test_emb.shape, test_labels.shape)
print("Full:", full_emb.shape, full_labels.shape)


# Output directory and file paths
csv_output_dir = os.path.join(DIR, f"MIA_results_{source}")
os.makedirs(csv_output_dir, exist_ok=True)

privacy_csv_path = os.path.join(csv_output_dir, f"SVC_MIA_training_privacy_{dataset_name}_{method}_m{n_model}.csv")
efficacy_csv_path = os.path.join(csv_output_dir, f"SVC_MIA_forget_efficacy_{dataset_name}_{method}_m{n_model}.csv")
MIA2_efficacy_csv_path = os.path.join(csv_output_dir, f"SVC_MIA2_forget_efficacy_{dataset_name}_{method}_m{n_model}.csv")

evaluation_metric_names = ["correctness", "confidence", "entropy", "m_entropy", "prob"]


privacy_columns = ["Forget Class", "Method", "Dataset", "Model", "n_model"] + evaluation_metric_names
efficacy_columns = ["Forget Class", "Method", "Dataset", "Model", "n_model"] + evaluation_metric_names
MIA2_columns = ["Forget Class", "Method", "Dataset", "Model", "n_model", "cv_score_mean", "cv_score_std"]


# Run once to create empty files with headers
pd.DataFrame(columns=privacy_columns).to_csv(privacy_csv_path, index=False)
pd.DataFrame(columns=efficacy_columns).to_csv(efficacy_csv_path, index=False)
pd.DataFrame(columns=MIA2_columns).to_csv(MIA2_efficacy_csv_path, index=False)

for forget_class in forget_classes:
    print(f"  - Forget class {forget_class}")
    

    if method == 'original':
        checkpoint_path_model = f"{DIR}/weights/chks_{dataset_name}/original/best_checkpoint_{model_name}_m{n_model}.pth"
        
    elif method == 'retrained':
        checkpoint_path_model = f"{DIR}/weights/chks_{dataset_name}/retrained/best_checkpoint_{model_name}_without_[{forget_class}].pth"
    
    else:
        checkpoint_path_model = f"{DIR}/checkpoints_main/{dataset_name}/{method}/samples_per_class_5000/resnet18_best_checkpoint_seed[{seed}]_class[{forget_class}]_m{n_model}_lr0.001.pt"


    model = get_model(model_name, dataset_name_upper, num_classes, checkpoint_path=checkpoint_path_model) 
  
    model.eval()
    fc_layer = model.fc.to(device)

    # Dataloaders using full data (not filtering forget class)
    train_loader = DataLoader(TensorDataset(train_emb, train_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_emb, test_labels), batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(TensorDataset(full_emb, full_labels), batch_size=batch_size, shuffle=False)


    # Forget set
    forget_mask_train = (train_labels == forget_class)
    forget_features_train = train_emb[forget_mask_train]
    forget_labels_train = train_labels[forget_mask_train]

    # Retain set
    retain_mask_train = (train_labels != forget_class)
    retain_features_train = train_emb[retain_mask_train]
    retain_labels_train = train_labels[retain_mask_train]
    
    # Forget set
    forget_mask_test = (test_labels == forget_class)
    forget_features_test = test_emb[forget_mask_test]
    forget_labels_test = test_labels[forget_mask_test]
    
    # Retain set
    retain_mask_test = (test_labels != forget_class)
    retain_features_test = test_emb[retain_mask_test]
    retain_labels_test = test_labels[retain_mask_test]
    
    # Forget set
    forget_mask_full = (full_labels == forget_class)
    forget_features_full = full_emb[forget_mask_full]
    forget_labels_full = full_labels[forget_mask_full]
    
    # Retain set
    retain_mask_full = (full_labels != forget_class)
    retain_features_full = full_emb[retain_mask_full]
    retain_labels_full = full_labels[retain_mask_full]
    
    # -------------------- Create DataLoaders for Forget and Retain Sets --------------------
    # Create TensorDatasets for each subset
    train_forget_dataset = TensorDataset(forget_features_train, forget_labels_train)
    train_retain_dataset = TensorDataset(retain_features_train, retain_labels_train)
    
    test_forget_dataset = TensorDataset(forget_features_test, forget_labels_test)
    test_retain_dataset = TensorDataset(retain_features_test, retain_labels_test)
    
    full_forget_dataset = TensorDataset(forget_features_full, forget_labels_full)
    full_retain_dataset = TensorDataset(retain_features_full, retain_labels_full)
    
    # Create DataLoader for each subset
    train_fgt_loader = DataLoader(train_forget_dataset, batch_size=batch_size, shuffle=True)
    train_retain_loader = DataLoader(train_retain_dataset, batch_size=batch_size, shuffle=True)
    
    test_fgt_loader = DataLoader(test_forget_dataset, batch_size=batch_size, shuffle=False)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=batch_size, shuffle=False)

    full_forget_loader = DataLoader(full_forget_dataset, batch_size=batch_size, shuffle=False)
    full_retain_loader = DataLoader(full_retain_dataset, batch_size=batch_size, shuffle=False)


    # Convert dataloaders to lists for indexing
    train_retain_data = list(train_retain_loader)
    test_data = list(test_loader)
    
    # Split into 50/50 for shadow and target
    split_train = len(train_retain_data) // 2
    split_test = len(test_data) // 2
    
    shadow_train_data = train_retain_data[:split_train]
    target_train_data = train_retain_data[split_train:]
    
    shadow_test_data = test_data[:split_test]
    target_test_data = test_data[split_test:]

    from torch.utils.data import DataLoader
    
    # Flatten the batches back into samples and rebuild datasets
    def flatten_batches(batched_data):
        inputs = []
        targets = []
        for x, y in batched_data:
            inputs.append(x)
            targets.append(y)
        return torch.cat(inputs), torch.cat(targets)
    
    # Create new datasets
    shadow_train_dataset = torch.utils.data.TensorDataset(*flatten_batches(shadow_train_data))
    target_train_dataset = torch.utils.data.TensorDataset(*flatten_batches(target_train_data))
    shadow_test_dataset = torch.utils.data.TensorDataset(*flatten_batches(shadow_test_data))
    target_test_dataset = torch.utils.data.TensorDataset(*flatten_batches(target_test_data))
    
    # Rebuild loaders
    batch_size = train_retain_loader.batch_size
    num_workers = train_retain_loader.num_workers
    
    shadow_train_loader_MIA_training_privacy = DataLoader(shadow_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_train_loader_MIA_training_privacy = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    shadow_test_loader_MIA_training_privacy = DataLoader(shadow_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    target_test_loader_MIA_training_privacy = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    



    # ------------------ SVC_MIA: Training Privacy ------------------
    mia_training_result = SVC_MIA(
        shadow_train=shadow_train_loader_MIA_training_privacy,
        shadow_test=shadow_test_loader_MIA_training_privacy,
        target_train=target_train_loader_MIA_training_privacy,
        target_test=target_test_loader_MIA_training_privacy,
        model=fc_layer,
    )
    
    row_privacy = {
        "Forget Class": forget_class,
        "Method": method,
        "Dataset": dataset_name,
        "Model": model_name,
        "n_model": n_model,
        **mia_training_result
    }
    pd.DataFrame([row_privacy]).to_csv(privacy_csv_path, mode='a', header=False, index=False)
    
    
    # ------------------ MIA1: SVC_MIA: Forget Efficacy ------------------
    mia_forget_result = SVC_MIA(
        shadow_train=train_retain_loader,
        shadow_test=test_loader,
        target_train=None,
        target_test=train_fgt_loader,
        model=fc_layer,
    )
    
    row_efficacy = {
        "Forget Class": forget_class,
        "Method": method,
        "Dataset": dataset_name,
        "Model": model_name,
        "n_model": n_model,
        **mia_forget_result
    }
    pd.DataFrame([row_efficacy]).to_csv(efficacy_csv_path, mode='a', header=False, index=False)


    # ------------------ MIA2 ------------------
    MIA2_forget_result = membership_inference_attack(
        model=fc_layer,
        t_loader=test_loader,
        f_loader=train_fgt_loader,
        seed=42,
    )
    
    # Wrap the score (which is a numpy array) into a dict
    MIA2_row_efficacy = {
        "Forget Class": forget_class,
        "Method": method,
        "Dataset": dataset_name,
        "Model": model_name,
        "n_model": n_model,
        "cv_score_mean": float(np.mean(MIA2_forget_result)),
        "cv_score_std": float(np.std(MIA2_forget_result))
    }

    pd.DataFrame([MIA2_row_efficacy]).to_csv(MIA2_efficacy_csv_path, mode='a', header=False, index=False)





