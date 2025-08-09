import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from generate_part_samples_randomly import RemainingResNet, TruncatedResNet
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
lr=0.0001



# File paths
DIR = "/projets/Zdehghani/MU_data_free"
#DIR = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"
checkpoint_dir = f"{DIR}/checkpoints_main_part/{dataset_name}/{method}/samples_per_class_{samples_per_class}"
embedding_train_file = f"{DIR}/tsne/tsne_main_part/{dataset_name}/{method}/real_embeddings_train_{dataset_name}_seed_{seed}_m{n_model}_n{samples_per_class}.npz"
embedding_test_file = f"{DIR}/tsne/tsne_main_part/{dataset_name}/{method}/real_embeddings_test_{dataset_name}_seed_{seed}_m{n_model}_n{samples_per_class}.npz"
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

print(original_model)
print(unlearned_model)



# Load embeddings
real_train_data = np.load(embedding_train_file)
real_test_data = np.load(embedding_test_file)
synth_data = np.load(synth_file)

print("real_train_data keys:", real_train_data.files)
print("real_test_data keys:", real_test_data.files)
print("synth_data keys:", synth_data.files)

real_embeddings_train_all = torch.tensor(real_train_data["real_embeddings"], dtype=torch.float32).to(device)
real_embeddings_test_all = torch.tensor(real_test_data["real_embeddings"], dtype=torch.float32).to(device)

real_labels_train_all = torch.tensor(real_train_data["real_labels"], dtype=torch.long).to(device)
real_labels_test_all = torch.tensor(real_test_data["real_labels"], dtype=torch.long).to(device)

print("real_embeddings_train shape:", real_embeddings_train_all.shape)
print("real_labels_train shape:", real_labels_train_all.shape)

print("real_embeddings_test shape:", real_embeddings_test_all.shape)
print("real_labels_test shape:", real_labels_test_all.shape)


real_embeddings_train_np, real_labels_train_np = select_n_per_class_numpy(
    real_embeddings_train_all.cpu(), real_labels_train_all.cpu(),
    num_per_class=N, num_classes=num_classes
)

real_embeddings_train = torch.tensor(real_embeddings_train_np, dtype=torch.float32).to(device)
real_labels_train = torch.tensor(real_labels_train_np, dtype=torch.long).to(device)


real_embeddings_test_np, real_labels_test_np = select_n_per_class_numpy(
    real_embeddings_test_all.cpu(), real_labels_test_all.cpu(),
    num_per_class=N, num_classes=num_classes
)

real_embeddings_test = torch.tensor(real_embeddings_test_np, dtype=torch.float32).to(device)
real_labels_test = torch.tensor(real_labels_test_np, dtype=torch.long).to(device)



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

real_train_probs_original = get_probs(original_model, real_embeddings_train)
real_train_probs_unlearned = get_probs(unlearned_model, real_embeddings_train)
real_test_probs_original = get_probs(original_model, real_embeddings_test)
real_test_probs_unlearned = get_probs(unlearned_model, real_embeddings_test)
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
    #plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)

    #plt.xlabel("t-SNE Dimension 1", fontsize=14)
    #plt.ylabel("t-SNE Dimension 2", fontsize=14)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

        
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
tsne_and_plot(real_train_probs_original, real_labels_train, "Real train - Original", "tsne_real_train_original_probs.png", forget_class)
tsne_and_plot(real_train_probs_unlearned, real_labels_train, "Real train - Unlearned", "tsne_real_train_unlearned_probs.png", forget_class)
tsne_and_plot(real_test_probs_original, real_labels_test, "Real test - Original", "tsne_real_original_probs.png", forget_class)
tsne_and_plot(real_test_probs_unlearned, real_labels_test, "Real test - Unlearned", "tsne_real_unlearned_probs.png", forget_class)


def get_logits(model, embeddings):
    with torch.no_grad():
        outputs = model(embeddings)
    return outputs  # raw logits


synth_logits_original = get_logits(original_model, synth_embeddings)
synth_logits_unlearned = get_logits(unlearned_model, synth_embeddings)
real_train_logits_original = get_logits(original_model, real_embeddings_train)
real_train_logits_unlearned = get_logits(unlearned_model, real_embeddings_train)
real_test_logits_original = get_logits(original_model, real_embeddings_test)
real_test_logits_unlearned = get_logits(unlearned_model, real_embeddings_test)


tsne_and_plot(synth_logits_original, synth_labels, "Synthetic - Original (Logits)", "tsne_synth_original_logits.png", forget_class)
tsne_and_plot(synth_logits_unlearned, synth_labels, "Synthetic - Unlearned (Logits)", "tsne_synth_unlearned_logits.png", forget_class)
tsne_and_plot(real_train_logits_original, real_labels_train, "Real train - Original (Logits)", "tsne_real_train_original_logits.png", forget_class)
tsne_and_plot(real_train_logits_unlearned, real_labels_train, "Real train - Unlearned (Logits)", "tsne_real_train_unlearned_logits.png", forget_class)
tsne_and_plot(real_test_logits_original, real_labels_test, "Real test - Original (Logits)", "tsne_real_test_original_logits.png", forget_class)
tsne_and_plot(real_test_logits_unlearned, real_labels_test, "Real test - Unlearned (Logits)", "tsne_real_test_unlearned_logits.png", forget_class)




synth_probs_original_np = synth_probs_original.cpu().numpy()
synth_probs_unlearned_np = synth_probs_unlearned.cpu().numpy()
synth_labels_np = synth_labels.cpu().numpy()

real_train_probs_original_np = real_train_probs_original.cpu().numpy()
real_train_probs_unlearned_np = real_train_probs_unlearned.cpu().numpy()
real_train_labels_np = real_labels_train.cpu().numpy()

real_test_probs_original_np = real_test_probs_original.cpu().numpy()
real_test_probs_unlearned_np = real_test_probs_unlearned.cpu().numpy()
real_test_labels_np = real_labels_test.cpu().numpy()


def filter_high_confidence(probs, labels, threshold=0.9):
    confidences, _ = torch.max(probs, dim=1)
    mask = confidences >= threshold
    return probs[mask], labels[mask]


# Filter high-confidence samples
highconf_probs_orig, highconf_labels_orig = filter_high_confidence(synth_probs_original, synth_labels, threshold=0.6)
highconf_probs_unlearned, highconf_labels_unlearned = filter_high_confidence(synth_probs_unlearned, synth_labels, threshold=0.6)

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

def load_truncated_and_remaining(model_path):
    from create_embeddings_utils import get_model
    from generate_part_samples_randomly import TruncatedResNet, RemainingResNet  # make sure TruncatedResNet is defined
    base_model = get_model("resnet18", dataset_name, num_classes, checkpoint_path=model_path).to(device)
    truncated = TruncatedResNet(base_model).to(device)
    remaining = RemainingResNet(base_model).to(device)
    truncated.eval()
    remaining.eval()
    return truncated, remaining





# ---- 1. Load truncated and remaining parts ----
from generate_part_samples_randomly import TruncatedResNet  # make sure it's defined
truncated_orig, remaining_orig = load_truncated_and_remaining(original_model_path)
truncated_unl, remaining_unl = load_truncated_and_remaining(unlearned_model_path)

# ---- 2. Load CIFAR-10 test images ----
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from torchvision import transforms

mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),

        }

std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        }



transform_train = {
    'cifar10': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['cifar10'], std=std['cifar10'])
    ]),
}

transform_test = {
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['cifar10'], std=std['cifar10'])
    ]),
}



# Select appropriate transformations
train_transform = transform_train.get(dataset_name, transform_test.get(dataset_name, None))
test_transform = transform_test.get(dataset_name, train_transform)


cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)



def get_probs_on_images(model, loader, truncated, remaining):
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            features = truncated(x)
            out = remaining(features)
            probs = torch.softmax(out, dim=1)
            all_probs.append(probs)
            all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)



cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
train_loader = DataLoader(cifar_train, batch_size=64, shuffle=False)






real_train_probs_orig_img, real_train_labels_orig_img = get_probs_on_images(original_model, train_loader, truncated_orig, remaining_orig)
real_test_probs_orig_img, real_test_labels_orig_img = get_probs_on_images(original_model, test_loader, truncated_orig, remaining_orig)

real_train_probs_unl_img, real_train_labels_unl_img = get_probs_on_images(unlearned_model, train_loader, truncated_unl, remaining_unl)
real_test_probs_unl_img, real_test_labels_unl_img = get_probs_on_images(unlearned_model, test_loader, truncated_unl, remaining_unl)






