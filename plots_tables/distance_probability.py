import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.stats import wasserstein_distance

from create_embeddings_utils import get_model

# ============================================================
# Configuration
# ============================================================
DIR = "/projets/Zdehghani/MU_data_free"
TSNE_ROOT = os.path.join(DIR, "tsne/tsne_main")
WEIGHTS_ROOT = os.path.join(DIR, "weights")

DATASET_NAME = "CIFAR10"   # "CIFAR10", "CIFAR100", or "TinyImageNet"
NUM_CLASSES = 10           # 10 for CIFAR10, 100 for CIFAR100, 200 for TinyImageNet
MODEL_NUM = 1              # m1
SEED = 42
N_SYNTH = 5000
BATCH_SIZE = 1024

ARCHS = ["resnet18", "resnet50", "swint", "ViT"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# MMD can get expensive if each class has many samples.
# This caps the number of samples per class used for MMD.
MMD_MAX_SAMPLES = 2000
MMD_EPS = 1e-12

if DATASET_NAME in ["CIFAR10", "CIFAR100"]:
    dataset_folder = DATASET_NAME.lower()
else:
    dataset_folder = DATASET_NAME


# ============================================================
# Helpers
# ============================================================
def get_classifier(model):
    for attr in ["heads", "fc", "head", "classifier", "classif"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        "Could not find classifier head. Expected one of: heads, fc, head, classifier, classif"
    )


def load_npz_embeddings(npz_path):
    data = np.load(npz_path)

    if "embeddings" in data.files and "labels" in data.files:
        embeddings = data["embeddings"].astype(np.float32)
        labels = data["labels"].astype(np.int64)

    elif "real_embeddings" in data.files and "real_labels" in data.files:
        embeddings = data["real_embeddings"].astype(np.float32)
        labels = data["real_labels"].astype(np.int64)

    elif "synthetic_embeddings" in data.files and "synthetic_labels" in data.files:
        embeddings = data["synthetic_embeddings"].astype(np.float32)
        labels = data["synthetic_labels"].astype(np.int64)

    else:
        raise KeyError(f"Unknown keys in {npz_path}. Found: {data.files}")

    return embeddings, labels.reshape(-1)


@torch.no_grad()
def compute_probabilities_from_embeddings(classifier, embeddings, batch_size=1024, device="cuda"):
    classifier.eval()
    probs_all = []

    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        x = torch.from_numpy(embeddings[start:end]).to(device)
        logits = classifier(x)
        probs = F.softmax(logits, dim=1)
        probs_all.append(probs.cpu().numpy())

    return np.concatenate(probs_all, axis=0)


# ============================================================
# Distance metrics
# ============================================================
def frechet_distance(X, Y, eps=1e-6):
    """
    FID-style Fréchet distance between two Gaussian fits of X and Y.
    X: [N1, d]
    Y: [N2, d]
    """
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return np.nan

    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)

    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    cov_x = cov_x + eps * np.eye(cov_x.shape[0])
    cov_y = cov_y + eps * np.eye(cov_y.shape[0])

    diff = mu_x - mu_y
    covmean, _ = linalg.sqrtm(cov_x @ cov_y, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    dist = diff @ diff + np.trace(cov_x + cov_y - 2 * covmean)
    return float(dist)


def _pairwise_sq_dists(X, Y):
    """
    Returns matrix [N, M] with squared Euclidean distances.
    """
    X_norm = np.sum(X * X, axis=1, keepdims=True)          # [N, 1]
    Y_norm = np.sum(Y * Y, axis=1, keepdims=True).T        # [1, M]
    D = X_norm + Y_norm - 2.0 * (X @ Y.T)
    return np.maximum(D, 0.0)


def _median_heuristic_sigma(X, Y, max_points=1000, seed=42):
    """
    Estimate RBF bandwidth from pooled samples.
    """
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X, Y], axis=0)

    if Z.shape[0] > max_points:
        idx = rng.choice(Z.shape[0], size=max_points, replace=False)
        Z = Z[idx]

    D = _pairwise_sq_dists(Z, Z)
    tri = D[np.triu_indices_from(D, k=1)]
    tri = tri[tri > 0]

    if len(tri) == 0:
        return 1.0

    sigma2 = np.median(tri)
    sigma = np.sqrt(max(sigma2, MMD_EPS))
    return float(sigma)


def mmd_rbf(X, Y, sigma=None, max_samples=2000, seed=42):
    """
    RBF-kernel MMD^2 between two sets of vectors.
    Uses subsampling to avoid huge memory/runtime when needed.
    """
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return np.nan

    rng = np.random.default_rng(seed)

    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]

    if Y.shape[0] > max_samples:
        idx = rng.choice(Y.shape[0], size=max_samples, replace=False)
        Y = Y[idx]

    if sigma is None:
        sigma = _median_heuristic_sigma(X, Y, seed=seed)

    gamma = 1.0 / (2.0 * sigma * sigma + MMD_EPS)

    Dxx = _pairwise_sq_dists(X, X)
    Dyy = _pairwise_sq_dists(Y, Y)
    Dxy = _pairwise_sq_dists(X, Y)

    Kxx = np.exp(-gamma * Dxx)
    Kyy = np.exp(-gamma * Dyy)
    Kxy = np.exp(-gamma * Dxy)

    # unbiased estimate
    n = X.shape[0]
    m = Y.shape[0]

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()

    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return float(max(mmd2, 0.0))


def wasserstein_mean_1d(X, Y):
    """
    Average 1D Wasserstein distance over dimensions.
    This is not the full multivariate Wasserstein distance.
    """
    if X.shape[0] < 1 or Y.shape[0] < 1:
        return np.nan

    d = X.shape[1]
    vals = []
    for j in range(d):
        vals.append(wasserstein_distance(X[:, j], Y[:, j]))
    return float(np.mean(vals))


def compute_all_distances(X, Y, seed=42):
    return {
        "frechet": frechet_distance(X, Y),
        "mmd": mmd_rbf(X, Y, sigma=None, max_samples=MMD_MAX_SAMPLES, seed=seed),
        "wasserstein": wasserstein_mean_1d(X, Y),
    }


# ============================================================
# Processing
# ============================================================
def process_architecture(arch_name):
    """
    For one architecture:
      1. load original checkpoint
      2. load real/synth embeddings
      3. compute probability vectors
      4. compute class-wise distances for all metrics
    """
    print(f"\n==================== {arch_name} ====================")

    ckpt_path = os.path.join(
        WEIGHTS_ROOT,
        f"chks_{dataset_folder}",
        "original",
        f"best_checkpoint_{arch_name}_m{MODEL_NUM}.pth"
    )

    arch_dir = os.path.join(TSNE_ROOT, arch_name, dataset_folder, "DELETE")

    real_path = os.path.join(
        arch_dir,
        f"real_embeddings_{dataset_folder}_seed_{SEED}_m{MODEL_NUM}_n{N_SYNTH}.npz"
    )
    synth_path = os.path.join(
        arch_dir,
        f"synth_embeddings_{dataset_folder}_seed_{SEED}_m{MODEL_NUM}_n{N_SYNTH}.npz"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found:\n{ckpt_path}")
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real embeddings not found:\n{real_path}")
    if not os.path.exists(synth_path):
        raise FileNotFoundError(f"Synthetic embeddings not found:\n{synth_path}")

    print(f"Checkpoint: {ckpt_path}")
    print(f"Real NPZ   : {real_path}")
    print(f"Synth NPZ  : {synth_path}")

    model = get_model(
        arch_name,
        DATASET_NAME,
        NUM_CLASSES,
        checkpoint_path=ckpt_path
    )
    classifier = get_classifier(model).to(device)
    classifier.eval()

    real_embeddings, real_labels = load_npz_embeddings(real_path)
    synth_embeddings, synth_labels = load_npz_embeddings(synth_path)

    print("real_embeddings :", real_embeddings.shape)
    print("synth_embeddings:", synth_embeddings.shape)

    real_probs = compute_probabilities_from_embeddings(
        classifier, real_embeddings, batch_size=BATCH_SIZE, device=device
    )
    synth_probs = compute_probabilities_from_embeddings(
        classifier, synth_embeddings, batch_size=BATCH_SIZE, device=device
    )

    print("real_probs :", real_probs.shape)
    print("synth_probs:", synth_probs.shape)

    metric_results = {
        "frechet": {},
        "mmd": {},
        "wasserstein": {},
    }

    for cls in range(NUM_CLASSES):
        real_mask = (real_labels == cls)
        synth_mask = (synth_labels == cls)

        real_cls_probs = real_probs[real_mask]
        synth_cls_probs = synth_probs[synth_mask]

        dists = compute_all_distances(real_cls_probs, synth_cls_probs, seed=SEED + cls)

        metric_results["frechet"][cls] = dists["frechet"]
        metric_results["mmd"][cls] = dists["mmd"]
        metric_results["wasserstein"][cls] = dists["wasserstein"]

        print(
            f"class {cls:3d} | "
            f"real={real_cls_probs.shape[0]:5d} | "
            f"synth={synth_cls_probs.shape[0]:5d} | "
            f"frechet={dists['frechet']:.6f} | "
            f"mmd={dists['mmd']:.6f} | "
            f"wasserstein={dists['wasserstein']:.6f}"
        )

    return metric_results


# ============================================================
# Main
# ============================================================
results_by_metric = {
    "frechet": {},
    "mmd": {},
    "wasserstein": {},
}

combined_rows = []

for arch in ARCHS:
    try:
        arch_results = process_architecture(arch)

        for metric_name in results_by_metric.keys():
            results_by_metric[metric_name][arch] = arch_results[metric_name]

            for cls, val in arch_results[metric_name].items():
                combined_rows.append({
                    "metric": metric_name,
                    "architecture": arch,
                    "class": cls,
                    "distance": val,
                })

    except Exception as e:
        print(f"\n[ERROR] {arch}: {e}")

        for metric_name in results_by_metric.keys():
            results_by_metric[metric_name][arch] = {
                cls: np.nan for cls in range(NUM_CLASSES)
            }
            for cls in range(NUM_CLASSES):
                combined_rows.append({
                    "metric": metric_name,
                    "architecture": arch,
                    "class": cls,
                    "distance": np.nan,
                })


# ============================================================
# Save one CSV per metric
# ============================================================
for metric_name, metric_dict in results_by_metric.items():
    df = pd.DataFrame(metric_dict)
    df.index.name = "class"
    df.loc["mean"] = df.mean(axis=0, skipna=True)

    out_csv = os.path.join(
        TSNE_ROOT,
        f"classwise_prob_{metric_name}_{dataset_folder}_seed{SEED}_m{MODEL_NUM}_n{N_SYNTH}.csv"
    )

    df.to_csv(out_csv)

    print(f"\nSaved {metric_name} CSV to:")
    print(out_csv)
    print(df)


# ============================================================
# Save combined long-format CSV
# ============================================================
combined_df = pd.DataFrame(combined_rows)
combined_out_csv = os.path.join(
    TSNE_ROOT,
    f"classwise_prob_all_metrics_{dataset_folder}_seed{SEED}_m{MODEL_NUM}_n{N_SYNTH}.csv"
)
combined_df.to_csv(combined_out_csv, index=False)

print("\nSaved combined CSV to:")
print(combined_out_csv)
print(combined_df.head())