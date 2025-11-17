import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import pandas as pd
import os
from create_embeddings_utils import get_model

DIR = "/projets/Zdehghani/MU_data_free"
weights_folder = "weights"
embeddings_folder = "embeddings"
model_name = 'resnet18'
num_forget_classes = 2
# -------------------- Configuration --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = {
    "CIFAR100": 100,
}

n_models = range(1, 6)
results = []

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Retrieve forget classes from the opt argument
def get_forget_classes(num_forget_classes, total_classes):
    # Define the permutation map (this can be replaced with any specific permutation)
    permutation_map = [25, 58, 38, 23, 96, 54, 51, 49, 98, 66,
                       16, 52, 40, 71, 63, 79, 53, 12, 46, 55,
                       83, 27, 41, 20, 30, 14, 70, 45, 61, 29,
                       4, 39, 21, 87, 60, 68, 75, 2, 92, 5,
                       57, 42, 0, 8, 97, 31, 50, 47, 13, 80,
                       34, 91, 17, 69, 85, 76, 94, 73, 99, 74,
                       43, 67, 62, 89, 36, 65, 26, 78, 19, 11,
                       90, 15, 3, 24, 72, 18, 33, 22, 7, 88,
                       44, 56, 86, 81, 82, 1, 48, 28, 6, 64,
                       9, 32, 35, 77, 95, 84, 59, 93, 10, 37]
    
    # Select the first `num_forget_classes` classes from the permutation map
    forget_classes = permutation_map[:num_forget_classes]  # Modify this if you want a more dynamic selection
    return forget_classes

for dataset_name, num_classes in datasets.items():
    for n_model in n_models:
        # Retrieve number of classes to forget from the opt configuration
        forget_classes = get_forget_classes(num_forget_classes, num_classes)


        if dataset_name.lower() in ["cifar10", "cifar100"]:
            dataset_name_upper = dataset_name.upper()
        else:
            dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"

        batch_size=256

        if dataset_name in ["CIFAR10", "CIFAR100"]:
            dataset_name_lower = dataset_name.lower()
        else:
            dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"


        # Load model and data as usual
        checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name.lower()}/original/best_checkpoint_{model_name}_m{n_model}.pth"
        model = get_model(model_name, dataset_name, num_classes, checkpoint_path=checkpoint_path_model)

        def get_classifier(model):
            if hasattr(model, "heads"):
                return model.heads
            if hasattr(model, "fc"):
                return model.fc
            raise AttributeError("Unknown classifier head: model has neither `heads` nor `fc`.")

        fc_layer = get_classifier(model)

        # Define dataset_name_upper
        dataset_name_upper = dataset_name.upper()

        # Load embeddings
        train_path = f"{DIR}/{embeddings_folder}/{dataset_name}/{model_name}_train_m{n_model}.npz"
        test_path = f"{DIR}/{embeddings_folder}/{dataset_name}/{model_name}_test_m{n_model}.npz"

        train_embeddings_data = np.load(train_path)
        test_embeddings_data = np.load(test_path)

        train_emb = torch.tensor(train_embeddings_data["embeddings"], dtype=torch.float32)
        train_labels = torch.tensor(train_embeddings_data["labels"], dtype=torch.long)
        test_emb = torch.tensor(test_embeddings_data["embeddings"], dtype=torch.float32)
        test_labels = torch.tensor(test_embeddings_data["labels"], dtype=torch.long)

        # Continue the rest of your code...


        # Create DataLoader for training data
        train_dataset = TensorDataset(train_emb, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        # Create DataLoader for testing data
        test_dataset = TensorDataset(test_emb, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # -------------------- Multi-Class Forget and Retain Sets --------------------
        forget_mask_train = np.isin(train_labels, forget_classes)
        forget_mask_test = np.isin(test_labels, forget_classes)

        retain_mask_train = ~forget_mask_train
        retain_mask_test = ~forget_mask_test

        forget_features_train = train_emb[forget_mask_train]
        forget_labels_train = train_labels[forget_mask_train]

        retain_features_train = train_emb[retain_mask_train]
        retain_labels_train = train_labels[retain_mask_train]

        forget_features_test = test_emb[forget_mask_test]
        forget_labels_test = test_labels[forget_mask_test]

        retain_features_test = test_emb[retain_mask_test]
        retain_labels_test = test_labels[retain_mask_test]

        # Create DataLoaders for Forget and Retain sets
        train_forget_dataset = TensorDataset(forget_features_train, forget_labels_train)
        train_retain_dataset = TensorDataset(retain_features_train, retain_labels_train)

        test_forget_dataset = TensorDataset(forget_features_test, forget_labels_test)
        test_retain_dataset = TensorDataset(retain_features_test, retain_labels_test)

        train_forget_loader = DataLoader(train_forget_dataset, batch_size=256, shuffle=True)
        train_retain_loader = DataLoader(train_retain_dataset, batch_size=256, shuffle=True)

        test_forget_loader = DataLoader(test_forget_dataset, batch_size=256, shuffle=False)
        test_retain_loader = DataLoader(test_retain_dataset, batch_size=256, shuffle=False)

        # Evaluate the model
        train_accuracy = evaluate_model(fc_layer, train_loader, device)
        print(f"Train Accuracy: {train_accuracy:.2f}%")

        test_accuracy = evaluate_model(fc_layer, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # Evaluate on the new forget/retain sets
        train_retain_acc = evaluate_model(fc_layer, train_retain_loader, device)
        train_fgt_acc = evaluate_model(fc_layer, train_forget_loader, device)
        val_test_retain_acc = evaluate_model(fc_layer, test_retain_loader, device)
        val_test_fgt_acc = evaluate_model(fc_layer, test_forget_loader, device)

        # Full Embeddings
        data_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/{model_name}_full_m{n_model}.npz"
        data = np.load(data_path)

        # Access full embeddings
        embeddings_full = data["embeddings"]  # Shape: (N, 512)
        labels_full = data["labels"]  # Shape: (N,)

        # Convert to tensors
        embeddings_tensor_full = torch.tensor(embeddings_full, dtype=torch.float32)
        labels_tensor_full = torch.tensor(labels_full, dtype=torch.long)

        # -------------------- Multi-Class Forget and Retain Sets --------------------
        forget_mask_full = np.isin(labels_tensor_full, forget_classes)
        retain_mask_full = ~forget_mask_full

        forget_embeddings_full = embeddings_tensor_full[forget_mask_full]
        forget_labels_full = labels_tensor_full[forget_mask_full]

        retain_embeddings_full = embeddings_tensor_full[retain_mask_full]
        retain_labels_full = labels_tensor_full[retain_mask_full]

        # Create DataLoaders for Full Forget and Retain sets
        full_forget_dataset = TensorDataset(forget_embeddings_full, forget_labels_full)
        full_retain_dataset = TensorDataset(retain_embeddings_full, retain_labels_full)

        full_forget_loader = DataLoader(full_forget_dataset, batch_size=256, shuffle=False)
        full_retain_loader = DataLoader(full_retain_dataset, batch_size=256, shuffle=False)

        # Evaluate on Full Forget and Retain sets
        full_retain_acc = evaluate_model(fc_layer, full_retain_loader, device)
        full_fgt_acc = evaluate_model(fc_layer, full_forget_loader, device)

        # Calculate the number of forget classes
        num_forget_classes_in_result = len(forget_classes)

        # After the evaluation on full forget and full retain sets

        # Calculate AUS (Area Under Score) for the test forget accuracy
        AUS = 1 / (1 + (val_test_fgt_acc / 100))  # You can adjust the formula if necessary
        print(f"AUS: {AUS:.3f}")

        # Convert forget_classes to a string of class indices joined by underscores
        forget_classes_str = "_".join(map(str, forget_classes))

        # Append the result
        result = {
            "Forget Class": forget_classes_str,  # Store the string of forget classes here
            "Num Forget Classes": num_forget_classes_in_result,
            "Mode": "original",
            "Dataset": dataset_name,
            "Model": model_name,
            "Model Num": n_model,
            "Train Acc": train_accuracy,
            "Test Acc": test_accuracy,
            "Train Retain Acc": train_retain_acc,
            "Train Forget Acc": train_fgt_acc,
            "Val Test Retain Acc": val_test_retain_acc,
            "Val Test Forget Acc": val_test_fgt_acc,
            "Val Full Retain Acc": full_retain_acc,
            "Val Full Forget Acc": full_fgt_acc,
            "AUS": AUS
        }

        # Append the result to the results list
        results.append(result)

        # Define the filename for the results CSV
        csv_filename = f"results_original_{model_name}.csv"

        # Save the results to CSV (append to file)
        pd.DataFrame([result]).to_csv(
            csv_filename, 
            mode="a", 
            header=not os.path.exists(csv_filename),  # Write header only if file doesn't exist
            index=False
        )
