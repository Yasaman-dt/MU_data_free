import torch
import numpy as np
import torchvision.models as models
from create_embeddings_utils import get_model
import os

model_name = "resnet18"
n_model = "3"

DATASET_NUM_CLASSES = {
    "CIFAR10": 10,
    #"STL10": 10,
    #"SVHN": 10,
    "CIFAR100": 100,
    #"Caltech101": 101,
    #"DTD": 47,
    #"Flowers102": 102,
    #"FGVCAircraft": 100,
    #"OxfordPets": 37,
    "TinyImageNet": 200,
}
    

DATASETS = [
    #'FGVCAircraft',
    #'Caltech101',
    #'OxfordPets',
    #'Flowers102',
    #'DTD',
    #'CIFAR10',
    #'STL10',
    #'SVHN',
    #'CIFAR100',
    'TinyImageNet'
]

DIR = "/projets/Zdehghani/MU_data_free"
weights_folder = "weights"
embeddings_folder = "embeddings"

# Function to extract convolution weights and compute embeddings
def process_dataset(dataset_name):
    num_classes = DATASET_NUM_CLASSES[dataset_name]
    
    if dataset_name in ["CIFAR10", "CIFAR100"]:
            dataset_name_lower = dataset_name.lower()
    else:
            dataset_name_lower = dataset_name  # keep original capitalization for "TinyImageNet"
                
    # Construct the checkpoint path
    checkpoint_path = f"{DIR}/{weights_folder}/chks_{dataset_name_lower}/original/best_checkpoint_resnet18_m{n_model}.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {dataset_name}: checkpoint not found at {checkpoint_path}")
        return
    
    model = get_model(model_name, dataset_name, num_classes, checkpoint_path=checkpoint_path) 
    
    
    # Define parameters
    input_size = 14  # Output of stage 3 is 14x14x256 (for 224x224 input)
    output_channels = 512  # Stage 4 has 512 filters
    
    # Get the first convolution layer in Stage 4
    conv_layer = model.layer4[0].conv1  # First conv in stage 4
    
    # Extract convolution weights
    conv_weights = conv_layer.weight.data  # Shape: (512, 256, 3, 3)
    
    # Get convolution parameters
    kernel_size = conv_layer.kernel_size[0]  # Expected: 3
    stride = conv_layer.stride[0]  # Expected: 2
    padding = conv_layer.padding[0]  # Check if non-zero
    
    
    output_size = (input_size - kernel_size + 2 * padding) // stride + 1
    
    print(f"Computed Output Size: {output_size} x {output_size}")
    
    
    # Convert convolution to matrix form (Im2Col)
    def get_conv_matrix(conv_weights):
        num_filters, in_channels, k_h, k_w = conv_weights.shape  # (512, 256, 3, 3)
        
        # Reshape each filter into a 1D vector
        B = conv_weights.view(num_filters, -1)  # Shape: (512, 2304)
        
        return B
    
    # Construct matrix B (corrected shape: (512, 2304))
    B_tensor = get_conv_matrix(conv_weights)
    
    # Convert to NumPy
    B_numpy = B_tensor.cpu().numpy()
    
    # Print the new shape of B
    print("Matrix B shape (NumPy):", B_numpy.shape)  # Expected: (512, 2304)
    
    # Save matrix B
    save_path = f"{DIR}/{weights_folder}/chks_{dataset_name_lower}/original/matrix_B_224_m{n_model}.npy"
    np.save(save_path, B_numpy)

    # -------------------------------------------------------------------
    
    # Load the saved B matrix
    load_path = f"{DIR}/{weights_folder}/chks_{dataset_name_lower}/original/matrix_B_224_m{n_model}.npy"
    B_numpy = np.load(load_path)
    B_tensor = torch.tensor(B_numpy, dtype=torch.float32)
    
    # Assume we have an example output from Stage 3 (batch size 1)
    X_stage3 = torch.randn(1, 256, 14, 14)  # Example input (batch_size=1)
    
    # Convert X_stage3 into patches (Im2Col transformation)
    unfold = torch.nn.Unfold(kernel_size=3, stride=2)
    X_patches = unfold(X_stage3)  # Shape: (1, 2304, 49) -> 49 patches of size 2304
    
    print("X_patches shape:", X_patches.shape)
    
    # Apply B (Matrix Multiplication)
    embeddings = torch.matmul(B_tensor, X_patches.squeeze(0))  # Shape: (512, 36)
    
    output_size = int(X_patches.shape[2] ** 0.5)  # Compute correct spatial size dynamically
    embeddings = embeddings.view(1, 512, output_size, output_size)  # Reshape to (1, 512, 6, 6)
    
    print("Final Embeddings Shape:", embeddings.shape)  # Expected: (1, 512, 6, 6)
    
    # Apply Global Average Pooling (GAP) to get a (1, 512) vector
    final_embedding = torch.mean(embeddings, dim=[2, 3])  # Averages over (6,6) spatial dimensions
    
    print("Final 512-Dimensional Feature Vector Shape:", final_embedding.shape)  # Expected: (1, 512)

    return final_embedding


# Process all datasets in the list
for dataset in DATASETS:
    try:
        process_dataset(dataset)
    except Exception as e:
        print(f"Error processing {dataset}: {e}")