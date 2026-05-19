import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from matplotlib.lines import Line2D

# Config
dataset_name = "cifar10"
n_model = 1
method = "RL"
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forget_class = 9
samples_per_class=5000
N=500
N_test=1000
lr=0.01



# File paths
DIR = "/projets/Zdehghani/MU_data_free"
#DIR = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"

checkpoint_dir = f"{DIR}/checkpoints_main/{dataset_name}/{method}/samples_per_class_{samples_per_class}"
embedding_file = f"{DIR}/tsne/tsne_main/{dataset_name}/real_embeddings_{dataset_name}_seed_{seed}_m{n_model}_n{N_test}.npz"
synth_file = f"{DIR}/tsne/tsne_main/{dataset_name}/synth_embeddings_{dataset_name}_seed_{seed}_m{n_model}_n{samples_per_class}.npz"
root_folder = f"{DIR}/tsne/tsne_main/{dataset_name}/{method}"  # new folder for saving plots
os.makedirs(f"{root_folder}/plots/class{forget_class}", exist_ok=True)

DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
}
num_classes = DATASET_NUM_CLASSES[dataset_name.lower()]

def load_reamaining(model_path):
    from create_embeddings_utils import get_model
    base_model = get_model("resnet18", dataset_name, num_classes, checkpoint_path=model_path).to(device)
    classifier = base_model.fc.to(device)
    classifier.eval()
    return classifier


#original_model_path = f"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/weights/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"
original_model_path = f"/projets/Zdehghani/MU_data_free/weights/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"
unlearned_model_path = os.path.join(checkpoint_dir, f"resnet18_best_checkpoint_seed[{seed}]_class[{forget_class}]_m{n_model}_lr{lr}.pt")

original_model = load_reamaining(original_model_path)
unlearned_model = load_reamaining(unlearned_model_path)

# Load embeddings
real_data = np.load(embedding_file)
synth_data = np.load(synth_file)

print("real_data keys:", real_data.files)
print("synth_data keys:", synth_data.files)



def select_n_per_class_numpy(embeddings, labels, num_per_class, num_classes):
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    embeddings = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

    selected_embeddings = []
    selected_labels = []

    for class_idx in range(num_classes):
        cls_indices = np.where(labels == class_idx)[0]
        if len(cls_indices) >= num_per_class:
            chosen_indices = np.random.choice(cls_indices, size=num_per_class, replace=False)
            selected_embeddings.append(embeddings[chosen_indices])
            selected_labels.append(labels[chosen_indices])
        else:
            print(f"Warning: Class {class_idx} has only {len(cls_indices)} samples")
    
    selected_embeddings = np.concatenate(selected_embeddings, axis=0)
    selected_labels = np.concatenate(selected_labels, axis=0)
    return selected_embeddings, selected_labels


real_embeddings_all = torch.tensor(real_data["real_embeddings"], dtype=torch.float32).to(device)
real_labels_all = torch.tensor(real_data["real_labels"], dtype=torch.long).to(device)

print("Loaded real_embeddings shape:", real_embeddings_all.shape)
print("Loaded real_labels shape:", real_labels_all.shape)


real_embeddings_np, real_labels_np = select_n_per_class_numpy(
    real_embeddings_all.cpu(), real_labels_all.cpu(),
    num_per_class=N, num_classes=num_classes
)

real_embeddings = torch.tensor(real_embeddings_np, dtype=torch.float32).to(device)
real_labels = torch.tensor(real_labels_np, dtype=torch.long).to(device)


print(f"Loaded {N} real_embeddings shape:", real_embeddings.shape)
print(f"Loaded {N} real_labels shape:", real_labels.shape)




synth_embeddings_all = torch.tensor(synth_data["synthetic_embeddings"], dtype=torch.float32).to(device)
synth_labels_all = torch.tensor(synth_data["synthetic_labels"], dtype=torch.long).to(device)

print("Loaded synth_embeddings shape:", synth_embeddings_all.shape)
print("Loaded synth_labels shape:", synth_labels_all.shape)


synth_embeddings_np, synth_labels_np = select_n_per_class_numpy(
    synth_embeddings_all.cpu(), synth_labels_all.cpu(),
    num_per_class=N, num_classes=num_classes
)

synth_embeddings = torch.tensor(synth_embeddings_np, dtype=torch.float32).to(device)
synth_labels = torch.tensor(synth_labels_np, dtype=torch.long).to(device)

print(f"Loaded {N} synth_embeddings shape:", synth_embeddings.shape)
print(f"Loaded {N} synth_labels shape:", synth_labels.shape)







# Forward pass
def get_probs(model, embeddings):
    with torch.no_grad():
        outputs = model(embeddings)
        probs = torch.softmax(outputs, dim=1)
    return probs

real_probs_original = get_probs(original_model, real_embeddings)
real_probs_unlearned = get_probs(unlearned_model, real_embeddings)
synth_probs_original = get_probs(original_model, synth_embeddings)
synth_probs_unlearned = get_probs(unlearned_model, synth_embeddings)


def tsne_and_plot(probs, labels, title, save_name, forget_class=9, show_legend=True):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(probs.cpu().numpy())

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    # Plot each class separately
    for class_idx in range(10):
        mask = (labels.cpu().numpy() == class_idx)
        label = f"{class_idx}" + (r"($c_f$)" if class_idx == forget_class else "")
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    color=cmap(class_idx),
                    label=label,
                    s=50)

    # Increase font sizes
    plt.title(title, fontsize=22)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels

    if show_legend:
        handles = [
            Line2D([0], [0], marker='o', color='w',
                   label=f"{i}" + (r"($c_f$)" if i == forget_class else ""),
                   markerfacecolor=cmap(i), markersize=8)
            for i in range(10)
        ]
        plt.legend(handles=handles, loc='best', fontsize=12, title='Class Name', title_fontsize=13)

    plt.tight_layout()
    plt.savefig(f"{root_folder}/plots/class{forget_class}/{save_name}", dpi=300)
    plt.close()



# def tsne_and_plot(probs, labels, title, save_name):
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     reduced = tsne.fit_transform(probs.cpu().numpy())

#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.cpu(), cmap="tab10", s=20)
#     plt.colorbar(scatter, ticks=range(10))
#     plt.title(title)
#     plt.xlabel("Dimension 1")
#     plt.ylabel("Dimension 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{root_folder}/plots/class{forget_class}/{save_name}", dpi=300)
#     plt.close()
    

# Run plots
tsne_and_plot(real_probs_original, real_labels, "Real - Original", "tsne_real_original_probs.png", forget_class)
tsne_and_plot(real_probs_unlearned, real_labels, "Real - Unlearned", "tsne_real_unlearned_probs.png", forget_class)
tsne_and_plot(synth_probs_original, synth_labels, "Synthetic - Original", "tsne_synth_original_probs.png", forget_class)
tsne_and_plot(synth_probs_unlearned, synth_labels, "Synthetic - Unlearned", "tsne_synth_unlearned_probs.png", forget_class)


def get_logits(model, embeddings):
    with torch.no_grad():
        outputs = model(embeddings)
    return outputs  # raw logits


real_logits_original = get_logits(original_model, real_embeddings)
real_logits_unlearned = get_logits(unlearned_model, real_embeddings)
synth_logits_original = get_logits(original_model, synth_embeddings)
synth_logits_unlearned = get_logits(unlearned_model, synth_embeddings)


tsne_and_plot(real_logits_original, real_labels, "Real - Original (Logits)", "tsne_real_original_logits.png", forget_class)
tsne_and_plot(real_logits_unlearned, real_labels, "Real - Unlearned (Logits)", "tsne_real_unlearned_logits.png", forget_class)
tsne_and_plot(synth_logits_original, synth_labels, "Synthetic - Original (Logits)", "tsne_synth_original_logits.png", forget_class)
tsne_and_plot(synth_logits_unlearned, synth_labels, "Synthetic - Unlearned (Logits)", "tsne_synth_unlearned_logits.png", forget_class)



# Save real tensors
real_probs_original_np = real_probs_original.cpu().numpy()
real_probs_unlearned_np = real_probs_unlearned.cpu().numpy()
real_labels_np = real_labels.cpu().numpy()


synth_probs_original_np = synth_probs_original.cpu().numpy()
synth_probs_unlearned_np = synth_probs_unlearned.cpu().numpy()
synth_labels_np = synth_labels.cpu().numpy()


def compute_accuracy(model, embeddings, labels, forget_class):
    with torch.no_grad():
        logits = model(embeddings)
        preds = torch.argmax(logits, dim=1)
        correct = preds.eq(labels)

        # Masks
        forget_mask = labels == forget_class
        retain_mask = labels != forget_class

        # Accuracy per group
        acc_all = correct.float().mean().item()
        acc_forget = correct[forget_mask].float().mean().item() if forget_mask.sum() > 0 else float('nan')
        acc_retain = correct[retain_mask].float().mean().item() if retain_mask.sum() > 0 else float('nan')

    return acc_all, acc_retain, acc_forget




# Compute accuracy on REAL embeddings
real_acc_o, real_retain_acc_o, real_forget_acc_o = compute_accuracy(original_model, real_embeddings, real_labels, forget_class)
real_acc_u, real_retain_acc_u, real_forget_acc_u = compute_accuracy(unlearned_model, real_embeddings, real_labels, forget_class)

# Compute accuracy on SYNTH embeddings
synth_acc_o, synth_retain_acc_o, synth_forget_acc_o = compute_accuracy(original_model, synth_embeddings, synth_labels, forget_class)
synth_acc_u, synth_retain_acc_u, synth_forget_acc_u = compute_accuracy(unlearned_model, synth_embeddings, synth_labels, forget_class)



print("\n=== REAL DATA ===")
print(f"Original   -> All: {real_acc_o:.4f}, Retain: {real_retain_acc_o:.4f}, Forget: {real_forget_acc_o:.4f}")
print(f"Unlearned  -> All: {real_acc_u:.4f}, Retain: {real_retain_acc_u:.4f}, Forget: {real_forget_acc_u:.4f}")

print("\n=== SYNTH DATA ===")
print(f"Original   -> All: {synth_acc_o:.4f}, Retain: {synth_retain_acc_o:.4f}, Forget: {synth_forget_acc_o:.4f}")
print(f"Unlearned  -> All: {synth_acc_u:.4f}, Retain: {synth_retain_acc_u:.4f}, Forget: {synth_forget_acc_u:.4f}")



