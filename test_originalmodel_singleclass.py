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
from torch.utils.data import TensorDataset, DataLoader
import copy
from create_embeddings_utils import get_model
import pandas as pd
import os

DIR = "/projets/Zdehghani/MU_data_free"
checkpoint_folder = "checkpoints"
weights_folder = "weights"
embeddings_folder = "embeddings"
model_name = 'ViT'

# -------------------- Configuration --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters for generated samples
#num_classes = 10
#dataset_name = 'CIFAR10'  # This can be dynamically selected

#num_classes = 100
#dataset_name = 'CIFAR100'  # This can be dynamically selected

#num_classes = 200
#dataset_name = 'TinyImageNet'  # This can be dynamically selected

#n_model = 2

    
# Construct the checkpoint path
#checkpoint_path = f"{DIR}/{checkpoint_folder}/{dataset_name}/{model_name}/checkpoint_classif1_run1.pth"
#checkpoint_path = f"best_checkpoint_resnet18.pth"

# Configuration
datasets = {
    #"CIFAR10": 10,
    "CIFAR100": 100,
    #"TinyImageNet": 200,
}

n_models = range(2, 3)  # example range: 1 to 2 for demonstration
results = []

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
            
for dataset_name, num_classes in datasets.items():
    for n_model in n_models:
        for forget_class in range(0, num_classes):  # just a subset for demo





            if dataset_name.lower() in ["cifar10", "cifar100"]:
                dataset_name_upper = dataset_name.upper()
            else:
                dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"

            batch_size=256

            if dataset_name in ["CIFAR10", "CIFAR100"]:
                dataset_name_lower = dataset_name.lower()
            else:
                dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"

            checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name_lower}/original/best_checkpoint_{model_name}_m{n_model}.pth"  # Set your actual checkpoint path
            model = get_model(model_name, dataset_name, num_classes, checkpoint_path=checkpoint_path_model) 

            def get_classifier(model):
                if hasattr(model, "heads"):
                    return model.heads
                if hasattr(model, "fc"):
                    return model.fc
                raise AttributeError("Unknown classifier head: model has neither `heads` nor `fc`.")

            # usage:
            fc_layer = get_classifier(model)



            # Define file paths for train and test embeddings and labels
            train_path = f"{DIR}/{embeddings_folder}/{dataset_name}/{model_name}_train_m{n_model}.npz"

            if dataset_name == "TinyImageNet":
                test_path = f"{DIR}/{embeddings_folder}/{dataset_name}/{model_name}_val_m{n_model}.npz"
            else:
                test_path = f"{DIR}/{embeddings_folder}/{dataset_name}/{model_name}_test_m{n_model}.npz"
            full_path = f"{DIR}/{embeddings_folder}/{dataset_name}/{model_name}_full_m{n_model}.npz"

                            
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


            data_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/{model_name}_full_m{n_model}.npz"

            data = np.load(data_path)
            embeddings_real = data["embeddings"]  # Shape: (N, 512)
            labels_real = data["labels"]  # Shape: (N,)

            print(f"Loaded Embeddings: {embeddings_real.shape}")

            # Convert to tensors
            embeddings_tensor_real = torch.tensor(embeddings_real, dtype=torch.float32)
            labels_tensor_real = torch.tensor(labels_real, dtype=torch.long)


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

            # Split into forget and retain sets
            forget_mask_real = labels_tensor_real == forget_class
            retain_mask_real = labels_tensor_real != forget_class

            # Forget set (samples from class 0)
            forget_embeddings_real = embeddings_tensor_real[forget_mask_real]
            forget_labels_real = labels_tensor_real[forget_mask_real]

            # Retain set (samples from all other classes)
            retain_embeddings_real = embeddings_tensor_real[retain_mask_real]
            retain_labels_real = labels_tensor_real[retain_mask_real]


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


            # Create DataLoaders for validation
            forgetfull_loader_real = DataLoader(TensorDataset(forget_embeddings_real, forget_labels_real), batch_size, shuffle=False)
            retainfull_loader_real = DataLoader(TensorDataset(retain_embeddings_real, retain_labels_real), batch_size, shuffle=False)


            # Evaluate on train retain set
            train_retain_acc = evaluate_model(fc_layer, train_retain_loader, device)
            print(f"Train Retain Accuracy: {train_retain_acc:.2f}%")

            # Evaluate on train forget set
            train_fgt_acc = evaluate_model(fc_layer, train_forget_loader, device)
            print(f"Train Forget Accuracy: {train_fgt_acc:.2f}%")

            # Evaluate on test retain set
            val_test_retain_acc = evaluate_model(fc_layer, test_retain_loader, device)
            print(f"Test Retain Accuracy: {val_test_retain_acc:.2f}%")

            # Evaluate on test forget set
            val_test_fgt_acc = evaluate_model(fc_layer, test_forget_loader, device)
            print(f"Test Forget Accuracy: {val_test_fgt_acc:.2f}%")

            # Evaluate on full retain set
            val_full_retain_acc = evaluate_model(fc_layer, retainfull_loader_real, device)
            print(f"Full Retain Accuracy: {val_full_retain_acc:.2f}%")

            # Evaluate on full forget set
            val_full_fgt_acc = evaluate_model(fc_layer, forgetfull_loader_real, device)
            print(f"Full Forget Accuracy: {val_full_fgt_acc:.2f}%")

            AUS = 1/(1+(val_test_fgt_acc/100))
            print(f"AUS: {AUS:.3f}")


            result = {
                "Forget Class": forget_class,
                "Mode": "original",
                "Dataset": dataset_name,
                "Model": model_name,
                "Model Num": n_model,
                "Train Acc": train_accuracy,
                "Test Acc": test_accuracy,
                "Train Retain Acc": train_retain_acc,
                "Train Forget Acc": train_fgt_acc,
                "Val Test Retain Acc": val_test_retain_acsc,
                "Val Test Forget Acc": val_test_fgt_acc,
                "Val Full Retain Acc": val_full_retain_acc,
                "Val Full Forget Acc": val_full_fgt_acc,
                "AUS": AUS,
            }
            results.append(result)
            
            pd.DataFrame([result]).to_csv(
                f"results_original_{model_name}.csv",
                mode="a",
                header=not os.path.exists(f"results_original_{model_name}.csv"),
                index=False
            )