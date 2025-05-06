import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from create_embeddings_utils import get_model
from torch.utils.data import DataLoader, TensorDataset


def generate_emb_samples(B_matrix, num_classes, samples_per_class, sigma_range, resnet_model, device='cuda'):
    """
    Generates synthetic feature embeddings and their corresponding labels using the given B matrix and passes them 
    through the last fully connected layer to obtain soft probabilities.

    Args:
        B_matrix (numpy.ndarray): The fully connected weight matrix (e.g., matrix_B_224.npy).
        num_classes (int): Number of classes.
        samples_per_class (int): Number of samples to generate per class.
        sigma_range (array-like): Range of sigma values for covariance scaling.
        resnet_model (torch.nn.Module): The pre-trained ResNet model for the last FC layer.
        device (str, optional): Device to run computations on. Defaults to 'cuda'.

    Returns:
        sample_features (torch.Tensor): Generated feature embeddings after pooling.
        sample_labels (torch.Tensor): Corresponding labels.
        probability_array (numpy.ndarray): Soft probabilities for each sample.
    """

    # Convert B_matrix to torch tensor
    weights = torch.tensor(B_matrix, dtype=torch.float32, device=device)  # Shape: (512, 2304)

    def generate_feature_samples(n_samples, Sigma, mean_vector, device):
        return torch.distributions.MultivariateNormal(
            torch.from_numpy(mean_vector).float().to(device),
            torch.from_numpy(Sigma).float().to(device)
        ).sample((n_samples,))

    # Compute correlation matrix R
    def compute_coefficient_matrix(weights):
        norm_weights = weights / torch.norm(weights, dim=1, keepdim=True)
        return torch.matmul(norm_weights, norm_weights.T)

    R = compute_coefficient_matrix(weights)

    def optimize_sigma(sigma_range):
        best_sigma, best_entropy, best_Sigma = sigma_range[0], float('inf'), None
        for sigma_val in sigma_range:
            sigma_values = torch.ones(R.shape[0], device=device) * sigma_val
            D = torch.diag(sigma_values)
            Sigma = torch.matmul(D, torch.matmul(R, D))
    
            mean_vector = np.zeros(R.shape[0])
            feature_samples = generate_feature_samples(100, Sigma.cpu().numpy(), mean_vector, device)
            
            # Ensure soft targets are valid
            entropy = -torch.sum(feature_samples * torch.log(feature_samples + 1e-10), dim=1).mean().item()
    
            if entropy < best_entropy or best_Sigma is None:  # Ensure at least one Sigma is selected
                best_entropy, best_sigma, best_Sigma = entropy, sigma_val, Sigma
        
        if best_Sigma is None:  # Ensure we always return a valid Sigma
            best_Sigma = torch.eye(R.shape[0], device=device) * 1  # Default to identity matrix scaled
    
        return best_sigma, best_Sigma

    best_sigma, best_Sigma = optimize_sigma(sigma_range)

    # Generate feature samples in the CNN feature map space (before pooling)
    total_samples = num_classes * samples_per_class
    mean_vector = np.zeros(best_Sigma.shape[0])

    sample_features = generate_feature_samples(total_samples, best_Sigma.cpu().numpy(), mean_vector, device)

    # Step 1: Reshape to CNN feature map dimensions (10000, 512, 6, 6) if possible
    output_size = 6  # Assuming feature map size is 6x6
    expected_feature_size = 512 * 6 * 6

    if best_Sigma.shape[0] == expected_feature_size:
        sample_features = sample_features.view(total_samples, 512, output_size, output_size)

        # Step 2: Apply Global Average Pooling (GAP) to reduce (512, 6, 6) → (512, 1, 1)
        sample_features = F.adaptive_avg_pool2d(sample_features, (1, 1))

        # Step 3: Flatten to (10000, 512)
        sample_features = sample_features.view(total_samples, -1)

    # Step 4: Pass through the last FC layer of pre-trained ResNet-18
    resnet_model.to(device)
    resnet_model.eval()
    with torch.no_grad():
        probability_array = F.softmax(resnet_model.fc(sample_features), dim=1)  # Shape: (10000, 10)

    # Step 5: Assign labels based on highest probability
    sample_labels = torch.argmax(probability_array, dim=1).to(device)

    return sample_features, sample_labels, probability_array.cpu().numpy()


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
        

## ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
#DIR = "/projets/Zdehghani/MU_data_free"
#checkpoint_folder = "checkpoints"
#weights_folder = "weights"
#embeddings_folder = "embeddings"
#files= "files"
#dataset_name = "cifar10"
#model_name = "resnet18"
#num_classes = 10
#n_model=1
#
#if dataset_name.lower() in ["cifar10", "cifar100"]:
#    dataset_name_upper = dataset_name.upper()
#else:
#    dataset_name_upper = dataset_name  # keep original capitalization for "tinyImagenet"
#
#
#checkpoint_path_model = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/best_checkpoint_resnet18_m{n_model}.pth"  # Set your actual checkpoint path
#model = get_model(model_name, dataset_name, num_classes, checkpoint_path=checkpoint_path_model) 
#
#
## Load the saved matrix_B_224
#matrix_B_224 = f"{DIR}/{weights_folder}/chks_{dataset_name}/original/matrix_B_224_m{n_model}.npy"
#
#B_numpy = np.load(matrix_B_224)  # Shape: (512, 2304)
#    
## Parameters
#samples_per_class = 1000  # Generate 1000 samples per class
#sigma_range = np.linspace(0.5, 5, 20)  # Range for sigma optimization
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
## Run the function
#sample_features, sample_labels, probability_array = generate_emb_samples(B_numpy, num_classes, samples_per_class, sigma_range, model, device=device)
#
## Print summary statistics
#print(f"Generated feature tensor shape: {sample_features.shape}")  # Expected: (10000, 512)
#print(f"Generated label tensor shape: {sample_labels.shape}")  # Expected: (10000,)
#print(f"Probability array shape: {probability_array.shape}")  # Expected: (10000, 10)
#print(f"Unique classes in generated labels: {torch.unique(sample_labels)}")
#
## Validate class distribution
#unique, counts = torch.unique(sample_labels, return_counts=True)
#class_distribution = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
#print(f"Class distribution: {class_distribution}")
#
## Validate feature statistics
#mean_features = sample_features.mean(dim=0)
#std_features = sample_features.std(dim=0)
#print(f"Feature mean: {mean_features.mean().item():.4f}, Feature std: {std_features.mean().item():.4f}")
#
## Check for NaNs or invalid values
#if torch.isnan(sample_features).any():
#    print("Warning: NaN values detected in generated features.")
#if torch.isnan(sample_labels).any():
#    print("Warning: NaN values detected in generated labels.")
#
#print("Feature generation test completed successfully!")
#
#
#
## ------------------ Load Real Embeddings ------------------
#
#train_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_train_m{n_model}.npz"
#test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_test_m{n_model}.npz"
#
#train_embeddings_data = np.load(train_path)
#test_embeddings_data = np.load(test_path)
#
#train_emb = torch.tensor(train_embeddings_data["embeddings"], dtype=torch.float32)
#train_labels = torch.tensor(train_embeddings_data["labels"], dtype=torch.long)
#
#test_emb = torch.tensor(test_embeddings_data["embeddings"], dtype=torch.float32)
#test_labels = torch.tensor(test_embeddings_data["labels"], dtype=torch.long)
#
## Move data to device
#train_emb, train_labels = train_emb.to(device), train_labels.to(device)
#test_emb, test_labels = test_emb.to(device), test_labels.to(device)
#
#batch_size = 256
#
## Create DataLoaders for Real and Synthetic Data
#train_loader = DataLoader(TensorDataset(train_emb, train_labels), batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(TensorDataset(test_emb, test_labels), batch_size=batch_size, shuffle=False)
#
#synth_train_loader = DataLoader(TensorDataset(sample_features, sample_labels), batch_size=batch_size, shuffle=True)
#
## ------------------ Evaluate Model Accuracy ------------------
#
#fc_layer = model.fc.to(device)
#
## Evaluate accuracy using synthetic embeddings
#synth_train_accuracy = evaluate_model(fc_layer, synth_train_loader, device)
#print(f"Synthetic Train Accuracy: {synth_train_accuracy:.2f}%")
#