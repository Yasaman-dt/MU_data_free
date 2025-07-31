import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from generate_part_samples_randomly import RemainingResNet

# Config
dataset_name = "cifar10"
n_model = 1
method = "NGFTW"
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forget_class = 0

# File paths
DIR = "/projets/Zdehghani/MU_data_free"
#DIR = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"

checkpoint_dir = f"{DIR}/checkpoints_main/{dataset_name}/{method}/samples_per_class_5000"
embedding_file = f"{DIR}/tsne/tsne_main/{dataset_name}/{method}/real_embeddings_{dataset_name}_seed_{seed}_m{n_model}_n50.npz"
synth_file = f"{DIR}/tsne/tsne_main/{dataset_name}/{method}/synth_embeddings_{dataset_name}_seed_{seed}_m{n_model}_n50.npz"
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
unlearned_model_path = os.path.join(checkpoint_dir, f"resnet18_best_checkpoint_seed[{seed}]_class[{forget_class}]_m{n_model}_lr0.001.pt")

original_model = load_reamaining(original_model_path)
unlearned_model = load_reamaining(unlearned_model_path)

# Load embeddings
real_data = np.load(embedding_file)
synth_data = np.load(synth_file)

print("real_data keys:", real_data.files)
print("synth_data keys:", synth_data.files)

real_embeddings = torch.tensor(real_data["real_embeddings"], dtype=torch.float32).to(device)
real_labels = torch.tensor(real_data["real_labels"], dtype=torch.long).to(device)

synthetic_embeddings = torch.tensor(synth_data["synthetic_embeddings"], dtype=torch.float32).to(device)
synthetic_labels = torch.tensor(synth_data["synthetic_labels"], dtype=torch.long).to(device)


# Forward pass
def get_probs(model, embeddings):
    with torch.no_grad():
        outputs = model(embeddings)
        probs = torch.softmax(outputs, dim=1)
    return probs

real_probs_original = get_probs(original_model, real_embeddings)
real_probs_unlearned = get_probs(unlearned_model, real_embeddings)
synth_probs_original = get_probs(original_model, synthetic_embeddings)
synth_probs_unlearned = get_probs(unlearned_model, synthetic_embeddings)

# t-SNE + Plot
def tsne_and_plot(probs, labels, title, save_name):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(probs.cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.cpu(), cmap="tab10", s=20)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{root_folder}/plots/class{forget_class}/{save_name}", dpi=300)
    plt.close()

# Run plots
tsne_and_plot(real_probs_original, real_labels, "Real - Original", "tsne_real_original_probs.png")
tsne_and_plot(real_probs_unlearned, real_labels, "Real - Unlearned", "tsne_real_unlearned_probs.png")
tsne_and_plot(synth_probs_original, synthetic_labels, "Synthetic - Original", "tsne_synth_original_probs.png")
tsne_and_plot(synth_probs_unlearned, synthetic_labels, "Synthetic - Unlearned", "tsne_synth_unlearned_probs.png")


def get_logits(model, embeddings):
    with torch.no_grad():
        outputs = model(embeddings)
    return outputs  # raw logits


real_logits_original = get_logits(original_model, real_embeddings)
real_logits_unlearned = get_logits(unlearned_model, real_embeddings)
synth_logits_original = get_logits(original_model, synthetic_embeddings)
synth_logits_unlearned = get_logits(unlearned_model, synthetic_embeddings)


tsne_and_plot(real_logits_original, real_labels, "Real - Original (Logits)", "tsne_real_original_logits.png")
tsne_and_plot(real_logits_unlearned, real_labels, "Real - Unlearned (Logits)", "tsne_real_unlearned_logits.png")
tsne_and_plot(synth_logits_original, synthetic_labels, "Synthetic - Original (Logits)", "tsne_synth_original_logits.png")
tsne_and_plot(synth_logits_unlearned, synthetic_labels, "Synthetic - Unlearned (Logits)", "tsne_synth_unlearned_logits.png")



# Save real tensors
real_probs_original_np = real_probs_original.cpu().numpy()
real_probs_unlearned_np = real_probs_unlearned.cpu().numpy()
real_labels_np = real_labels.cpu().numpy()


synth_probs_original_np = synth_probs_original.cpu().numpy()
synth_probs_unlearned_np = synth_probs_unlearned.cpu().numpy()
synth_labels_np = synthetic_labels.cpu().numpy()
