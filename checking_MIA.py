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
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser(description="Run SVC MIA pipeline with model and method options.")
parser.add_argument('--n_model', type=int, default=1, help='Model index (e.g., 1 to 5)')
parser.add_argument('--method', type=str, default='original', help='Unlearning method to evaluate (e.g., original, retrained, FT, etc.)')
parser.add_argument('--model_name', type=str, default='resnet18', help='Model name (e.g., resnet18, vit, etc.)')
args = parser.parse_args()

n_model = args.n_model
method = args.method
model_name = args.model_name


# ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
DIR = "/projets/Zdehghani/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"
dataset_name = "cifar10"
#model_name = "resnet18"
num_classes = 10
forget_classes = list(range(num_classes))  # or a subset if needed
source = "real" #synth
#n_model=1
batch_size = 1024
#method="original"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if dataset_name.lower() in ["cifar10", "cifar100"]:
    dataset_name_upper = dataset_name.upper()
else:
    dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"
    
if dataset_name in ["cifar10", "cifar100"]:
    dataset_name_lower = dataset_name.lower()
else:
    dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"


mean = {
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4867, 0.4408),
        'TinyImageNet': (0.485, 0.456, 0.406),
        }

std = {
        'CIFAR10': (0.2023, 0.1994, 0.2010),
        'CIFAR100': (0.2675, 0.2565, 0.2761),
        'TinyImageNet': (0.229, 0.224, 0.225),
        }



transform_train = {
    'CIFAR10': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['CIFAR10'], std=std['CIFAR10'])
    ]),
    'CIFAR100': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['CIFAR100'], std=std['CIFAR100'])
    ]),
    'TinyImageNet': transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['TinyImageNet'], std=std['TinyImageNet'])
    ])
}

transform_test = {
    'CIFAR10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['CIFAR10'], std=std['CIFAR10'])
    ]),
    'CIFAR100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['CIFAR100'], std=std['CIFAR100'])
    ]),
    'TinyImageNet': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['TinyImageNet'], std=std['TinyImageNet'])
    ])
}

transform_val = {
    'TinyImageNet': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['TinyImageNet'], std=std['TinyImageNet'])
    ])
}

# Select appropriate transformations
train_transform = transform_train.get(dataset_name_upper, transform_test.get(dataset_name_upper, None))
test_transform = transform_test.get(dataset_name_upper, train_transform)
val_transform = transform_val.get(dataset_name_upper, train_transform)


if dataset_name_lower == "cifar10":
    train_dataset = datasets.CIFAR10(root="./data/CIFAR10", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=test_transform)
    
if dataset_name_lower == "cifar100":
    train_dataset = datasets.CIFAR100(root="./data/CIFAR100", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root="./data/CIFAR100", train=False, download=True, transform=test_transform)

if dataset_name_lower == "TinyImageNet":
    train_dir = os.path.join("./data/TinyImageNet", "train")
    val_dir = os.path.join("./data/TinyImageNet", "val")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

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


if dataset_name_lower == "cifar10":
    train_dataset = datasets.CIFAR10(root="./data/CIFAR10", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=test_transform)
    
if dataset_name_lower == "cifar100":
    train_dataset = datasets.CIFAR100(root="./data/CIFAR100", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root="./data/CIFAR100", train=False, download=True, transform=test_transform)

if dataset_name_lower == "TinyImageNet":
    train_dir = os.path.join("./data/TinyImageNet", "train")
    val_dir = os.path.join("./data/TinyImageNet", "val")


    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(val_dir, transform=val_transform)



def extract_embeddings_through_model_from_dataset(model, dataset, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings, all_labels = [], []
    
    model.eval()
    model.to(device)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats = feature_extractor(inputs).squeeze()
            if feats.ndim == 1:
                feats = feats.unsqueeze(0)
            all_embeddings.append(feats.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_embeddings), torch.cat(all_labels)





# Output directory and file paths
csv_output_dir = os.path.join(DIR, f"MIA_results_{source}")
os.makedirs(csv_output_dir, exist_ok=True)

privacy_csv_path = os.path.join(csv_output_dir, f"SVC_MIA_training_privacy_{dataset_name}_{method}_m{n_model}.csv")
efficacy_csv_path = os.path.join(csv_output_dir, f"SVC_MIA_forget_efficacy_{dataset_name}_{method}_m{n_model}.csv")

evaluation_metric_names = ["correctness", "confidence", "entropy", "m_entropy", "prob"]


privacy_columns = ["Forget Class", "Method", "Dataset", "Model", "n_model"] + evaluation_metric_names
efficacy_columns = ["Forget Class", "Method", "Dataset", "Model", "n_model"] + evaluation_metric_names


# Run once to create empty files with headers
pd.DataFrame(columns=privacy_columns).to_csv(privacy_csv_path, index=False)
pd.DataFrame(columns=efficacy_columns).to_csv(efficacy_csv_path, index=False)

for forget_class in forget_classes:
    print(f"  - Forget class {forget_class}")
    

    if method == 'original':
        checkpoint_path_model = f"{DIR}/weights/chks_{dataset_name}/original/best_checkpoint_{model_name}_m{n_model}.pth"
        
    elif method == 'retrained':
        checkpoint_path_model = f"{DIR}/weights/chks_{dataset_name}/retrained/best_checkpoint_{model_name}_without_[{forget_class}].pth"
    
    else:
        checkpoint_path_model = f"{DIR}/out_{source}/samples_per_class_5000/CR/{dataset_name}/{method}/models/unlearned_model_{method}_m{n_model}_seed_42_class_{forget_class}.pth"


    model = get_model(model_name, dataset_name_upper, num_classes, checkpoint_path=checkpoint_path_model) 
  
    model.eval()
    fc_layer = model.fc.to(device)



    # Replace this block only for the retrained method
    if method == 'retrained':
        train_emb, train_labels = extract_embeddings_through_model_from_dataset(model, train_dataset, device)
        test_emb, test_labels = extract_embeddings_through_model_from_dataset(model, test_dataset, device)
        full_emb, full_labels = extract_embeddings_through_model_from_dataset(model, full_dataset, device)

    else:
        # Original loading from precomputed embeddings
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
    
    
    # ------------------ SVC_MIA: Forget Efficacy ------------------
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

