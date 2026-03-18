import os
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import wasserstein_distance

# ============================================================
# Configuration
# ============================================================
DIR = "/projets/Zdehghani/MU_data_free"
TSNE_ROOT = os.path.join(DIR, "tsne/tsne_main")

DATASET_NAME = "CIFAR10"   # "CIFAR10", "CIFAR100", or "TinyImageNet"
NUM_CLASSES = 10           # 10 / 100 / 200
MODEL_NUM = 1              # m1
SEED = 42
N_SYNTH = 5000
METHOD = "DELETE"          # folder name inside tsne_main

ARCHS = ["resnet18", "resnet50", "swint", "ViT"]

# MMD is expensive if there are many samples
MMD_MAX_SAMPLES = 2000
MMD_EPS = 1e-12

if DATASET_NAME in ["CIFAR10", "CIFAR100"]:
    dataset_folder = DATASET_NAME.lower()
else:
    dataset_folder = DATASET_NAME


# ============================================================
# NPZ loader
# ============================================================
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


# ============================================================
# Distance metrics
# ============================================================
def frechet_distance(X, Y, eps=1e-6):
    """
    Fréchet distance between two Gaussian fits of X and Y.
    X: [N1, d]
    Y: [N2, d]
    """
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return np.nan

    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)

    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    cov_x = cov_x + eps * np.eye(cov_x.shape[0], dtype=np.float64)
    cov_y = cov_y + eps * np.eye(cov_y.shape[0], dtype=np.float64)

    diff = mu_x - mu_y
    covmean, _ = linalg.sqrtm(cov_x @ cov_y, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    dist = diff @ diff + np.trace(cov_x + cov_y - 2 * covmean)
    return float(dist)


def _pairwise_sq_dists(X, Y):
    X_norm = np.sum(X * X, axis=1, keepdims=True)            # [N, 1]
    Y_norm = np.sum(Y * Y, axis=1, keepdims=True).T          # [1, M]
    D = X_norm + Y_norm - 2.0 * (X @ Y.T)
    return np.maximum(D, 0.0)


def _median_heuristic_sigma(X, Y, max_points=1000, seed=42):
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
    Unbiased RBF-kernel MMD^2.
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
    Average 1D Wasserstein distance over embedding dimensions.
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
# Per-architecture processing
# ============================================================
def process_architecture(arch_name):
    print(f"\n==================== {arch_name} ====================")

    arch_dir = os.path.join(TSNE_ROOT, arch_name, dataset_folder, METHOD)

    real_path = os.path.join(
        arch_dir,
        f"real_embeddings_{dataset_folder}_seed_{SEED}_m{MODEL_NUM}_n{N_SYNTH}.npz"
    )
    synth_path = os.path.join(
        arch_dir,
        f"synth_embeddings_{dataset_folder}_seed_{SEED}_m{MODEL_NUM}_n{N_SYNTH}.npz"
    )

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real embeddings not found:\n{real_path}")
    if not os.path.exists(synth_path):
        raise FileNotFoundError(f"Synthetic embeddings not found:\n{synth_path}")

    print(f"Real NPZ : {real_path}")
    print(f"Synth NPZ: {synth_path}")

    real_embeddings, real_labels = load_npz_embeddings(real_path)
    synth_embeddings, synth_labels = load_npz_embeddings(synth_path)

    print("real_embeddings :", real_embeddings.shape)
    print("synth_embeddings:", synth_embeddings.shape)

    metric_results = {
        "frechet": {},
        "mmd": {},
        "wasserstein": {},
    }

    for cls in range(NUM_CLASSES):
        real_mask = (real_labels == cls)
        synth_mask = (synth_labels == cls)

        X_real = real_embeddings[real_mask]
        X_synth = synth_embeddings[synth_mask]

        dists = compute_all_distances(X_real, X_synth, seed=SEED + cls)

        metric_results["frechet"][cls] = dists["frechet"]
        metric_results["mmd"][cls] = dists["mmd"]
        metric_results["wasserstein"][cls] = dists["wasserstein"]

        print(
            f"class {cls:3d} | "
            f"real={X_real.shape[0]:5d} | "
            f"synth={X_synth.shape[0]:5d} | "
            f"dim={X_real.shape[1] if X_real.shape[0] > 0 else 'NA'} | "
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
        f"classwise_embedding_{metric_name}_{dataset_folder}_seed{SEED}_m{MODEL_NUM}_n{N_SYNTH}.csv"
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
    f"classwise_embedding_all_metrics_{dataset_folder}_seed{SEED}_m{MODEL_NUM}_n{N_SYNTH}.csv"
)

combined_df.to_csv(combined_out_csv, index=False)

print("\nSaved combined CSV to:")
print(combined_out_csv)
print(combined_df.head())