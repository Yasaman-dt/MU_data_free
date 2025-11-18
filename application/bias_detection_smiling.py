import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import os

# Modify the model to include two classification heads
class DualHeadResNet18(nn.Module):
    def __init__(self):
        super(DualHeadResNet18, self).__init__()
        # Use pre-trained ResNet18 without the fully connected (classification) layer
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last fully connected layer (classifier)
        
        # We assume 512 as the feature size from ResNet18 after the adaptive average pooling
        self.fc_input_features = 512
        
        # New fully connected layer to extract features
        self.fc = nn.Linear(self.fc_input_features, 512)  # 512 as the output feature size
        
        # Smiling classification head (binary: Smiling vs Not Smiling)
        self.smiling_head = nn.Linear(512, 1)
        
        # Gender classification head (binary: Male vs Female)
        self.gender_head = nn.Linear(512, 2)
    
    def forward(self, x):
        # Forward pass through ResNet layers
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        # Pass through the fully connected layer
        x = self.fc(x)
        
        # Pass through the Smiling head
        smiling_output = self.smiling_head(x)
        
        # Pass through the Gender head
        gender_output = self.gender_head(x)
        
        return smiling_output, gender_output


# Set device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset Setup (CelebA)
data_dir = './data'  # You can specify your desired directory to store the data

# Transformations (Standard CelebA transformations)
transform_test = transforms.Compose([
    transforms.Resize(178),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean and std
])

# Download and load the CelebA dataset for testing
test_dataset = datasets.CelebA(root=data_dir, split='test', transform=transform_test, download=True)

# Create DataLoader for batch processing
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load your trained model
model = DualHeadResNet18()  # Assuming you've trained it using the DualHeadResNet18 class
model.load_state_dict(torch.load('best_dual_head_model_smiling.pth'))
model = model.to(device)

# Set the model to evaluation mode
model.eval()

def evaluate_bias(model, test_loader, smiling_idx=31, gender_idx=20):
    correct_smiling_female = 0
    correct_smiling_male = 0
    correct_non_smiling_female = 0
    correct_non_smiling_male = 0

    total_smiling_female = 0
    total_smiling_male = 0
    total_non_smiling_female = 0
    total_non_smiling_male = 0

    with torch.no_grad():
        for inputs, full_labels in test_loader:
            smiling_labels = full_labels[:, smiling_idx]
            gender_labels = full_labels[:, gender_idx]
            inputs, smiling_labels, gender_labels = inputs.to(device), smiling_labels.to(device), gender_labels.to(device)

            smiling_output, gender_output = model(inputs)
            predicted_smiling = torch.round(torch.sigmoid(smiling_output)).squeeze()
            _, predicted_gender = torch.max(gender_output, 1)

            # Smiling and Female
            smiling_female_mask = (smiling_labels == 1) & (gender_labels == 0)
            correct_smiling_female += predicted_smiling[smiling_female_mask].eq(smiling_labels[smiling_female_mask]).sum().item()
            total_smiling_female += smiling_female_mask.sum().item()

            # Smiling and Male
            smiling_male_mask = (smiling_labels == 1) & (gender_labels == 1)
            correct_smiling_male += predicted_smiling[smiling_male_mask].eq(smiling_labels[smiling_male_mask]).sum().item()
            total_smiling_male += smiling_male_mask.sum().item()

            # Non-Smiling and Female
            non_smiling_female_mask = (smiling_labels == 0) & (gender_labels == 0)
            correct_non_smiling_female += predicted_smiling[non_smiling_female_mask].eq(smiling_labels[non_smiling_female_mask]).sum().item()
            total_non_smiling_female += non_smiling_female_mask.sum().item()

            # Non-Smiling and Male
            non_smiling_male_mask = (smiling_labels == 0) & (gender_labels == 1)
            correct_non_smiling_male += predicted_smiling[non_smiling_male_mask].eq(smiling_labels[non_smiling_male_mask]).sum().item()
            total_non_smiling_male += non_smiling_male_mask.sum().item()

    # Avoid division by zero
    smiling_accuracy_female = (correct_smiling_female / total_smiling_female * 100) if total_smiling_female > 0 else 0
    smiling_accuracy_male = (correct_smiling_male / total_smiling_male * 100) if total_smiling_male > 0 else 0
    non_smiling_accuracy_female = (correct_non_smiling_female / total_non_smiling_female * 100) if total_non_smiling_female > 0 else 0
    non_smiling_accuracy_male = (correct_non_smiling_male / total_non_smiling_male * 100) if total_non_smiling_male > 0 else 0

    total_smiling = total_smiling_female + total_smiling_male
    total_non_smiling = total_non_smiling_female + total_non_smiling_male
    total_female = total_smiling_female + total_non_smiling_female
    total_male = total_smiling_male + total_non_smiling_male

    correct_smiling = correct_smiling_female + correct_smiling_male
    correct_non_smiling = correct_non_smiling_female + correct_non_smiling_male
    correct_female = correct_smiling_female + correct_non_smiling_female
    correct_male = correct_smiling_male + correct_non_smiling_male

    # 1) accuracy on smiling (ignore gender)
    smiling_acc_overall = (correct_smiling / total_smiling * 100) if total_smiling > 0 else 0
    # 2) accuracy on non-smiling (ignore gender)
    non_smiling_acc_overall = (correct_non_smiling / total_non_smiling * 100) if total_non_smiling > 0 else 0
    # 3) accuracy on females (ignore smiling)
    female_acc_overall = (correct_female / total_female * 100) if total_female > 0 else 0
    # 4) accuracy on males (ignore smiling)
    male_acc_overall = (correct_male / total_male * 100) if total_male > 0 else 0

    print(f"Smiling Females: {total_smiling_female}")
    print(f"Smiling Males: {total_smiling_male}")
    print(f"Non-Smiling Females: {total_non_smiling_female}")
    print(f"Non-Smiling Males: {total_non_smiling_male}")

    print(f"Smiling Accuracy for Females: {smiling_accuracy_female:.2f}%")
    print(f"Smiling Accuracy for Males: {smiling_accuracy_male:.2f}%")
    print(f"Non-Smiling Accuracy for Females: {non_smiling_accuracy_female:.2f}%")
    print(f"Non-Smiling Accuracy for Males: {non_smiling_accuracy_male:.2f}%")

    print(f"\n[Ignoring gender]")
    print(f"Smiling Accuracy (overall): {smiling_acc_overall:.2f}%")
    print(f"Non-Smiling Accuracy (overall): {non_smiling_acc_overall:.2f}%")

    print(f"\n[Ignoring smiling]")
    print(f"Female Accuracy (overall): {female_acc_overall:.2f}%")
    print(f"Male Accuracy (overall): {male_acc_overall:.2f}%")

    if abs(smiling_accuracy_female - smiling_accuracy_male) > 10:
        print("\nPotential Bias Detected in Smiling Prediction")
    else:
        print("\nNo Significant Bias Detected in Smiling Prediction")

# Test the bias in predictions
evaluate_bias(model, test_loader)
