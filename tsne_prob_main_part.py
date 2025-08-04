import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from generate_part_samples_randomly import RemainingResNet
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
lr=0.001



# File paths
DIR = "/projets/Zdehghani/MU_data_free"
#DIR = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"
checkpoint_dir = f"{DIR}/checkpoints_main_part/{dataset_name}/{method}/samples_per_class_{samples_per_class}"
#embedding_file = f"{DIR}/tsne/tsne_main_part/{dataset_name}/{method}/real_embeddings_{dataset_name}_seed_{seed}_m{n_model}_n{samples_per_class}.npz"
synth_file = f"{DIR}/tsne/tsne_main_part/{dataset_name}/{method}/synth_embeddings_{dataset_name}_seed_{seed}_m{n_model}_n{samples_per_class}.npz"
root_folder = f"{DIR}/tsne/tsne_main_part/{dataset_name}/{method}"  # new folder for saving plots
os.makedirs(f"{root_folder}/plots/class{forget_class}", exist_ok=True)


DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
}
num_classes = DATASET_NUM_CLASSES[dataset_name.lower()]

def load_reamaining(model_path):
    from create_embeddings_utils import get_model
    base_model = get_model("resnet18", dataset_name, num_classes, checkpoint_path=model_path).to(device)
    classifier = RemainingResNet(base_model).to(device)
    classifier.eval()
    return classifier


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


#original_model_path = f"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free/weights/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"
original_model_path = f"/projets/Zdehghani/MU_data_free/weights/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"
unlearned_model_path = os.path.join(checkpoint_dir, f"resnet18_best_checkpoint_seed[{seed}]_class[{forget_class}]_m{n_model}_lr{lr}.pt")

original_model = load_reamaining(original_model_path)
unlearned_model = load_reamaining(unlearned_model_path)

# Load embeddings
#real_data = np.load(embedding_file)
synth_data = np.load(synth_file)

#print("real_data keys:", real_data.files)
print("synth_data keys:", synth_data.files)

#real_embeddings = torch.tensor(real_data["real_embeddings"], dtype=torch.float32).to(device)
#real_labels = torch.tensor(real_data["real_labels"], dtype=torch.long).to(device)

#print("real_embeddings shape:", real_embeddings.shape)
#print("real_labels shape:", real_labels.shape)


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

# Forward pass
def get_probs(model, embeddings):
    with torch.no_grad():
        outputs = model(embeddings)
        probs = torch.softmax(outputs, dim=1)
    return probs

#real_probs_original = get_probs(original_model, real_embeddings)
#real_probs_unlearned = get_probs(unlearned_model, real_embeddings)
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
    #plt.title(title, fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)

    #plt.xticks([])  # Remove x-axis tick labels
    #plt.yticks([])  # Remove y-axis tick labels

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




# Run plots
tsne_and_plot(synth_probs_original, synth_labels, "Synthetic - Original", "tsne_synth_original_probs.png", forget_class)
tsne_and_plot(synth_probs_unlearned, synth_labels, "Synthetic - Unlearned", "tsne_synth_unlearned_probs.png", forget_class)


def get_logits(model, embeddings):
    with torch.no_grad():
        outputs = model(embeddings)
    return outputs  # raw logits


synth_logits_original = get_logits(original_model, synth_embeddings)
synth_logits_unlearned = get_logits(unlearned_model, synth_embeddings)


tsne_and_plot(synth_logits_original, synth_labels, "Synthetic - Original (Logits)", "tsne_synth_original_logits.png", forget_class)
tsne_and_plot(synth_logits_unlearned, synth_labels, "Synthetic - Unlearned (Logits)", "tsne_synth_unlearned_logits.png", forget_class)



synth_probs_original_np = synth_probs_original.cpu().numpy()
synth_probs_unlearned_np = synth_probs_unlearned.cpu().numpy()
synth_labels_np = synth_labels.cpu().numpy()



def filter_high_confidence(probs, labels, threshold=0.9):
    confidences, _ = torch.max(probs, dim=1)
    mask = confidences >= threshold
    return probs[mask], labels[mask]


# Filter high-confidence samples
highconf_probs_orig, highconf_labels_orig = filter_high_confidence(synth_probs_original, synth_labels, threshold=0.9)
highconf_probs_unlearned, highconf_labels_unlearned = filter_high_confidence(synth_probs_unlearned, synth_labels, threshold=0.9)

# Plot t-SNE for high-confidence samples
tsne_and_plot(highconf_probs_orig, highconf_labels_orig, "High-Confidence Synthetic - Original", "tsne_highconf_synth_original_probs.png", forget_class)
tsne_and_plot(highconf_probs_unlearned, highconf_labels_unlearned, "High-Confidence Synthetic - Unlearned", "tsne_highconf_synth_unlearned_probs.png", forget_class)



highconf_labels_orig_np = highconf_labels_orig.cpu().numpy()
highconf_probs_orig_np = highconf_probs_orig.cpu().numpy()
highconf_labels_unlearned_np = highconf_labels_unlearned.cpu().numpy()
highconf_probs_unlearned_np = highconf_probs_unlearned.cpu().numpy()



orig_counts = torch.bincount(highconf_labels_orig, minlength=num_classes)
unlearned_counts = torch.bincount(highconf_labels_unlearned, minlength=num_classes)

min_counts_each = torch.minimum(orig_counts, unlearned_counts)


global_min = min_counts_each.min().item()


def sample_fixed_per_class(probs, labels, global_min, num_classes):
    selected_probs = []
    selected_labels = []

    for class_name in range(num_classes):
        # Get indices for the current class
        cls_indices = (labels == class_name).nonzero(as_tuple=True)[0]
        if len(cls_indices) >= global_min:
            # Shuffle and select global_min
            selected = cls_indices[torch.randperm(len(cls_indices))[:global_min]]
            selected_probs.append(probs[selected])
            selected_labels.append(labels[selected])
        else:
            print(f"Warning: class {class_name} has only {len(cls_indices)} samples, less than global_min={global_min}")

    # Concatenate all selected
    return torch.cat(selected_probs), torch.cat(selected_labels)

num_classes = 10  # or detect from data

balanced_probs_orig, balanced_labels_orig = sample_fixed_per_class(
    highconf_probs_orig, highconf_labels_orig, global_min, num_classes
)

balanced_probs_unlearned, balanced_labels_unlearned = sample_fixed_per_class(
    highconf_probs_unlearned, highconf_labels_unlearned, global_min, num_classes
)

print("Balanced shapes:", balanced_probs_orig.shape, balanced_labels_orig.shape)


tsne_and_plot(balanced_probs_orig, balanced_labels_orig, "T-SNE of Softmax Probabilities Before Unlearning", "tsne_balanced_highconf_original.png", forget_class, show_legend=False)

tsne_and_plot(balanced_probs_unlearned, balanced_labels_unlearned, "T-SNE of Softmax Probabilities After Unlearning", "tsne_balanced_highconf_unlearned.png", forget_class, show_legend=False)


balanced_synth_embeddings, balanced_synth_labels = sample_fixed_per_class(
    synth_embeddings, synth_labels, global_min, num_classes
)

print("Balanced synth shape:", balanced_synth_embeddings.shape)


balanced_probs_orig_np = balanced_probs_orig.cpu().numpy()
balanced_labels_orig_np = balanced_labels_orig.cpu().numpy()
balanced_probs_unlearned_np = balanced_probs_unlearned.cpu().numpy()
balanced_labels_unlearned_np = balanced_labels_unlearned.cpu().numpy()



if balanced_synth_embeddings.ndim > 2:
    balanced_synth_embeddings_flat = balanced_synth_embeddings.view(balanced_synth_embeddings.size(0), -1)
else:
    balanced_synth_embeddings_flat = balanced_synth_embeddings

tsne_and_plot(balanced_synth_embeddings_flat, balanced_synth_labels, "T-SNE of Synthetic Embeddings", "tsne_balanced_synth_embeddings.png", forget_class, show_legend=False)


