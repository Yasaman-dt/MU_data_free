import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from SCRUB_data_free import SCRUB_data_free
from generate_emb_samples_resnet18 import generate_emb_samples
from torch.utils.data import TensorDataset, DataLoader
import copy
from create_embeddings_utils import get_model

DIR = "/projets/Zdehghani/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"

# -------------------- Configuration --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

forget_class = 0  

# Parameters for generated samples
#num_classes = 10
#dataset_name = 'CIFAR10'  # This can be dynamically selected

#num_classes = 100
#dataset_name = 'CIFAR100'  # This can be dynamically selected

num_classes = 200
dataset_name = 'TinyImageNet'  # This can be dynamically selected

n_model = 2
model_name = 'resnet18'   # This can also be dynamically selected

batch_size=256

if dataset_name in ["CIFAR10", "CIFAR100"]:
    dataset_name_lower = dataset_name.lower()
else:
    dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"
    
# Construct the checkpoint path
#checkpoint_path = f"{DIR}/{checkpoint_folder}/{dataset_name}/{model_name}/checkpoint_classif1_run1.pth"
#checkpoint_path = f"best_checkpoint_resnet18.pth"

checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name_lower}/original/best_checkpoint_resnet18_m{n_model}.pth"  # Set your actual checkpoint path
model = get_model(model_name, dataset_name, num_classes, checkpoint_path=checkpoint_path_model) 

fc_layer = model.fc
 
print(fc_layer)



# Define file paths for train and test embeddings and labels
train_path = f"{DIR}/{embeddings_folder}/{dataset_name}/resnet18_train_m{n_model}.npz"

if dataset_name == "TinyImageNet":
    test_path = f"{DIR}/{embeddings_folder}/{dataset_name}/resnet18_val_m{n_model}.npz"
else:
    test_path = f"{DIR}/{embeddings_folder}/{dataset_name}/resnet18_test_m{n_model}.npz"
full_path = f"{DIR}/{embeddings_folder}/{dataset_name}/resnet18_full_m{n_model}.npz"

                
train_embeddings_data = np.load(train_path)
test_embeddings_data = np.load(test_path)


# Access the embeddings and labels
train_emb = train_embeddings_data["embeddings"]  # The embeddings for the training data
train_labels = train_embeddings_data["labels"]   # The labels for the training data

test_emb = test_embeddings_data["embeddings"]  # The embeddings for the training data
test_labels = test_embeddings_data["labels"]   # The labels for the training data

print("Train embeddings shape:", train_emb.shape)
print("Test embeddings shape:", test_emb.shape)

train_emb = torch.tensor(train_emb, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)


test_emb = torch.tensor(test_emb, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Create a DataLoader for training data
train_dataset = TensorDataset(train_emb, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create a DataLoader for testing data
test_dataset = TensorDataset(test_emb, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



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


# Move model to the correct device
fc_layer.to(device)

# Evaluate on the training set
train_accuracy = evaluate_model(fc_layer, train_loader, device)
print(f"Train Accuracy: {train_accuracy:.2f}%")

# Evaluate on the test set
test_accuracy = evaluate_model(fc_layer, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")


# -------------------- Separate Forget and Retain Sets --------------------
# Forget set: only one class
forget_mask_train = (train_labels == forget_class)
forget_features_train = train_emb[forget_mask_train]
forget_labels_train = train_labels[forget_mask_train]

# Retain set: all other classes
retain_mask_train = (train_labels != forget_class)
retain_features_train = train_emb[retain_mask_train]
retain_labels_train = train_labels[retain_mask_train]

# Forget set: only one class in test set
forget_mask_test = (test_labels == forget_class)
forget_features_test = test_emb[forget_mask_test]
forget_labels_test = test_labels[forget_mask_test]

# Retain set: all other classes in test set
retain_mask_test = (test_labels != forget_class)
retain_features_test = test_emb[retain_mask_test]
retain_labels_test = test_labels[retain_mask_test]

# -------------------- Create DataLoaders for Forget and Retain Sets --------------------
# Create TensorDatasets for each subset
train_forget_dataset = TensorDataset(forget_features_train, forget_labels_train)
train_retain_dataset = TensorDataset(retain_features_train, retain_labels_train)

test_forget_dataset = TensorDataset(forget_features_test, forget_labels_test)
test_retain_dataset = TensorDataset(retain_features_test, retain_labels_test)

# Create DataLoader for each subset
train_forget_loader = DataLoader(train_forget_dataset, batch_size=batch_size, shuffle=True)
train_retain_loader = DataLoader(train_retain_dataset, batch_size=batch_size, shuffle=True)

test_forget_loader = DataLoader(test_forget_dataset, batch_size=batch_size, shuffle=False)
test_retain_loader = DataLoader(test_retain_dataset, batch_size=batch_size, shuffle=False)


# Evaluate on train retain set
train_retain_accuracy = evaluate_model(fc_layer, train_retain_loader, device)
print(f"Train Retain Accuracy: {train_retain_accuracy:.2f}%")

# Evaluate on train forget set
train_forget_accuracy = evaluate_model(fc_layer, train_forget_loader, device)
print(f"Train Forget Accuracy: {train_forget_accuracy:.2f}%")

# Evaluate on test retain set
test_retain_accuracy = evaluate_model(fc_layer, test_retain_loader, device)
print(f"Test Retain Accuracy: {test_retain_accuracy:.2f}%")

# Evaluate on test forget set
test_forget_accuracy = evaluate_model(fc_layer, test_forget_loader, device)
print(f"Test Forget Accuracy: {test_forget_accuracy:.2f}%")