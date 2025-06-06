import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# ---------------- Settings ----------------
DIR = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/checkingforgetting"
embeddings_folder = "embeddings"
dataset_name = "cifar10"
dataset_name_upper = dataset_name.upper()
n_model = 1
num_classes = 10
forget_classes = list(range(num_classes))
use_logits = True
input_type = "logits" if use_logits else "probs"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

methods = ["FT", "NG", "NGFTW", "RE", "SCAR", "SCRUB"]

results = []

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
    
    
for method in methods:
    print(f"\n=== Evaluating method: {method} ===")

    # ---------------- Gather Train and Test ----------------
    train_logits_list, train_real_labels_list, train_forget_flags = [], [], []
    test_logits_list, test_real_labels_list, test_forget_flags = [], [], []

    for forget_class in forget_classes:
        base_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/{method}/logits_forget_class_{forget_class}"
        
        train_data = np.load(f"{base_path}/train_logits_m{n_model}.npz")
        test_data = np.load(f"{base_path}/test_logits_m{n_model}.npz")

        # Select input type
        train_logits_list.append(train_data["logits"] if use_logits else train_data["probs"])
        test_logits_list.append(test_data["logits"] if use_logits else test_data["probs"])

        train_real_labels_list.append(train_data["real_labels"])
        train_forget_flags.append(train_data["forget_labels"])

        test_real_labels_list.append(test_data["real_labels"])
        test_forget_flags.append(test_data["forget_labels"])

        # Concatenate
        train_logits = np.concatenate(train_logits_list, axis=0)
        train_labels = np.concatenate(train_real_labels_list, axis=0)
        test_logits = np.concatenate(test_logits_list, axis=0)
        test_labels = np.concatenate(test_real_labels_list, axis=0)
    
        # ---------------- PyTorch Dataset and Loader ----------------
        train_X = torch.tensor(train_logits, dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        train_set = TensorDataset(train_X, train_y)
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    
        test_X = torch.tensor(test_logits, dtype=torch.float32)
        test_y = torch.tensor(test_labels, dtype=torch.long)
        test_set = TensorDataset(test_X, test_y)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    
        # ---------------- MLP Classifier ----------------

        model = MLPClassifier(train_X.shape[1], num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
    
        best_test_acc = 0
        patience = 20
        counter = 0
    
        for epoch in range(300):
            # ---- Train ----
            model.train()
            total_train_loss = 0
            total_train_samples = 0
            all_train_preds, all_train_true = [], []
    
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
    
                total_train_loss += loss.item()
                total_train_samples += X_batch.size(0)
                all_train_preds.append(preds.argmax(dim=1).detach().cpu())
                all_train_true.append(y_batch.cpu())
    
            avg_train_loss = total_train_loss / total_train_samples
    
            train_preds = torch.cat(all_train_preds)
            train_true = torch.cat(all_train_true)
            train_acc = accuracy_score(train_true, train_preds)
    
            # ---- Test ----
            model.eval()
            all_test_preds, all_test_true = [], []
            total_test_loss = 0
            total_test_samples = 0
    
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    loss = loss_fn(preds, y_batch)
                    total_test_loss += loss.item()
                    total_test_samples += X_batch.size(0)
                    all_test_preds.append(preds.argmax(dim=1).cpu())
                    all_test_true.append(y_batch.cpu())
    
            avg_test_loss = total_test_loss / total_test_samples
    
            test_preds = torch.cat(all_test_preds)
            test_true = torch.cat(all_test_true)
            test_acc = accuracy_score(test_true, test_preds)
    
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
                    break
    
            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Best Test Acc: {best_test_acc:.4f} | No improvement: {counter}/{patience}")
        
        # Save result for current method
        results.append({
            "method": method,
            "dataset": dataset_name,
            "forget_class": forget_class,
            "input_type": input_type,
            "best_test_acc": best_test_acc
        })

# ---------------- Write All Results to CSV ----------------
results_df = pd.DataFrame(results)
csv_path = f"{DIR}/results_summary_perclassforgetting.csv"

if os.path.exists(csv_path):
    results_df.to_csv(csv_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(csv_path, index=False)

print(f"\n✅ Saved all results to {csv_path}")
