import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from create_embeddings_utils import get_model
from sklearn.manifold import TSNE

def generate_emb_samples_balanced(num_classes, samples_per_class, sigma_range, model, device='cuda'):

    # Function to compute coefficient matrix R from weights
    def compute_coefficient_matrix(weights):
        weights = weights.reshape(weights.shape[0], -1)
        norm_weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        return np.dot(norm_weights, norm_weights.T)


    # Function to generate feature space samples
    def generate_feature_samples(n_samples, Sigma, mean_vector, device='cuda'):
        feature_samples = torch.distributions.MultivariateNormal(
            torch.from_numpy(mean_vector).float().to(device),
            torch.from_numpy(Sigma).float().to(device)
        ).sample((n_samples,))
        return feature_samples

    # Function to generate soft targets from feature space samples
    def generate_soft_targets(model, feature_samples, layer_index, temperature=1.0, device='cuda'):
        with torch.no_grad():
            if layer_index == -1:  # Directly using generated feature samples
                soft_targets = F.softmax(feature_samples / temperature, dim=1)
                predicted_labels = torch.argmax(soft_targets, dim=1)   # shape: (n_samples,)
        return soft_targets, predicted_labels


    def accumulate_per_class_samples(fc_layer, Sigma, mean_vector, device, 
                                    n_classes, n_per_class, 
                                    batch_size=1000, threshold=0.97):
        class_counts = {i: 0 for i in range(n_classes)}
        class_features = {i: [] for i in range(n_classes)}
        class_soft_targets = {i: [] for i in range(n_classes)}
        
        layer_index = -1
        while any(class_counts[c] < n_per_class for c in range(n_classes)):
            # Step 1: Generate a batch of synthetic features
            feature_samples = generate_feature_samples(batch_size, Sigma, mean_vector, device)
            soft_targets, predicted_labels = generate_soft_targets(fc_layer, feature_samples, layer_index, device=device)

            confidence_scores = torch.max(soft_targets, dim=1).values

            print("synth_labels shape:", predicted_labels.shape)

            high_conf_mask = confidence_scores > threshold

            high_conf_features = feature_samples[high_conf_mask]
            high_conf_targets = soft_targets[high_conf_mask]
            high_conf_labels = predicted_labels[high_conf_mask]

            # Step 2: Accumulate up to n_per_class per class
            for i in range(high_conf_features.shape[0]):
                class_name = int(high_conf_labels[i].item())
                if class_counts[class_name] < n_per_class:
                    class_features[class_name].append(high_conf_features[i].unsqueeze(0))
                    class_soft_targets[class_name].append(high_conf_targets[i].unsqueeze(0))
                    class_counts[class_name] += 1

            print(f"Current class counts: {class_counts}")

        # Step 3: Stack all features and soft targets
        all_features = torch.cat([torch.cat(class_features[c], dim=0) for c in range(n_classes)], dim=0)
        all_targets = torch.cat([torch.cat(class_soft_targets[c], dim=0) for c in range(n_classes)], dim=0)
        all_labels = torch.cat([torch.full((n_per_class,), c, dtype=torch.long) for c in range(n_classes)], dim=0)

        return all_features, all_targets, all_labels


    if isinstance(model.fc, nn.Sequential):
        for module in model.fc:
            if isinstance(module, nn.Linear):
                fc_layer = module
                break
    else:
        fc_layer = model.fc

    fc_layer = fc_layer.to(device)

    weights = fc_layer.weight.detach().cpu().numpy()
    R = compute_coefficient_matrix(weights)
    sigma_val = 50
    sigma_values = np.ones(R.shape[0]) * sigma_val
    D = np.diag(sigma_values)
    Sigma = np.dot(D, np.dot(R, D))
    mean_vector = np.zeros(R.shape[0])
    num_per_class = samples_per_class

    synthetic_features, synthetic_soft_targets, synthetic_labels = accumulate_per_class_samples(
        fc_layer=fc_layer,
        Sigma=Sigma,
        mean_vector=mean_vector,
        device=device,
        n_classes=num_classes,
        n_per_class=num_per_class,  # total will be 1000
        batch_size=1000,
        threshold=0.97)



    embedding_dim = 512
    lr = 0.01
    num_iterations = 3000
    temperature = 1
    n_samples = num_per_class * num_classes

    # === Step 2: Initialize embeddings to be optimized ===
    optimized_embeddings = torch.randn(n_samples, embedding_dim, requires_grad=True, device=device)

    # === Step 3: Optimize embeddings to match soft targets ===
    optimizer = torch.optim.Adam([optimized_embeddings], lr=lr)
    loss_fn = nn.KLDivLoss(reduction='batchmean')

    for step in range(num_iterations):
        optimizer.zero_grad()

        logits_pred = fc_layer(optimized_embeddings)
        log_probs = F.log_softmax(logits_pred / temperature, dim=1)

        synthetic_soft_targets = synthetic_soft_targets.to(device)

        loss = loss_fn(log_probs, synthetic_soft_targets)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")


    optimized_embeddings_np = optimized_embeddings.detach().cpu().numpy()

    return optimized_embeddings_np, synthetic_labels, synthetic_soft_targets


