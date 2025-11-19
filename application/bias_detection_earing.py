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
        
        # earing classification head (binary: earing vs Not earing)
        self.smiling_head = nn.Linear(512, 1)
        
        # Gender classification head (binary: Male vs Female)
        self.gender_head = nn.Linear(512, 2)
    
    def forward(self, x):
        # Forward pass through ResNet layers
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        # Pass through the fully connected layer
        x = self.fc(x)
        
        # Pass through the earing head
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
model.load_state_dict(torch.load('best_dual_head_model_earing.pth'))
model = model.to(device)

# Set the model to evaluation mode
model.eval()

def evaluate_bias(model, test_loader, earing_idx=34, gender_idx=20):
    correct_earing_female = 0
    correct_earing_male = 0
    correct_non_earing_female = 0
    correct_non_earing_male = 0

    total_earing_female = 0
    total_earing_male = 0
    total_non_earing_female = 0
    total_non_earing_male = 0

    with torch.no_grad():
        for inputs, full_labels in test_loader:
            earing_labels = full_labels[:, earing_idx]
            gender_labels = full_labels[:, gender_idx]
            inputs, earing_labels, gender_labels = inputs.to(device), earing_labels.to(device), gender_labels.to(device)

            earing_output, gender_output = model(inputs)
            predicted_earing = torch.round(torch.sigmoid(earing_output)).squeeze()
            _, predicted_gender = torch.max(gender_output, 1)

            # earing and Female
            earing_female_mask = (earing_labels == 1) & (gender_labels == 0)
            correct_earing_female += predicted_earing[earing_female_mask].eq(earing_labels[earing_female_mask]).sum().item()
            total_earing_female += earing_female_mask.sum().item()

            # earing and Male
            earing_male_mask = (earing_labels == 1) & (gender_labels == 1)
            correct_earing_male += predicted_earing[earing_male_mask].eq(earing_labels[earing_male_mask]).sum().item()
            total_earing_male += earing_male_mask.sum().item()

            # Non-earing and Female
            non_earing_female_mask = (earing_labels == 0) & (gender_labels == 0)
            correct_non_earing_female += predicted_earing[non_earing_female_mask].eq(earing_labels[non_earing_female_mask]).sum().item()
            total_non_earing_female += non_earing_female_mask.sum().item()

            # Non-earing and Male
            non_earing_male_mask = (earing_labels == 0) & (gender_labels == 1)
            correct_non_earing_male += predicted_earing[non_earing_male_mask].eq(earing_labels[non_earing_male_mask]).sum().item()
            total_non_earing_male += non_earing_male_mask.sum().item()

    # Avoid division by zero
    earing_accuracy_female = (correct_earing_female / total_earing_female * 100) if total_earing_female > 0 else 0
    earing_accuracy_male = (correct_earing_male / total_earing_male * 100) if total_earing_male > 0 else 0
    non_earing_accuracy_female = (correct_non_earing_female / total_non_earing_female * 100) if total_non_earing_female > 0 else 0
    non_earing_accuracy_male = (correct_non_earing_male / total_non_earing_male * 100) if total_non_earing_male > 0 else 0

    total_earing = total_earing_female + total_earing_male
    total_non_earing = total_non_earing_female + total_non_earing_male
    total_female = total_earing_female + total_non_earing_female
    total_male = total_earing_male + total_non_earing_male

    correct_earing = correct_earing_female + correct_earing_male
    correct_non_earing = correct_non_earing_female + correct_non_earing_male
    correct_female = correct_earing_female + correct_non_earing_female
    correct_male = correct_earing_male + correct_non_earing_male

    # 1) accuracy on earing (ignore gender)
    earing_acc_overall = (correct_earing / total_earing * 100) if total_earing > 0 else 0
    # 2) accuracy on non-earing (ignore gender)
    non_earing_acc_overall = (correct_non_earing / total_non_earing * 100) if total_non_earing > 0 else 0
    # 3) accuracy on females (ignore earing)
    female_acc_overall = (correct_female / total_female * 100) if total_female > 0 else 0
    # 4) accuracy on males (ignore earing)
    male_acc_overall = (correct_male / total_male * 100) if total_male > 0 else 0

    print(f"Males earing: {total_earing_male}")
    print(f"Females earing: {total_earing_female}")
    print(f"Males Non-earing: {total_non_earing_male}")
    print(f"Females Non-earing: {total_non_earing_female}")

    print(f"earing Accuracy for Males: {earing_accuracy_male:.2f}%")
    print(f"earing Accuracy for Females: {earing_accuracy_female:.2f}%")
    print(f"Non-earing Accuracy for Males: {non_earing_accuracy_male:.2f}%")
    print(f"Non-earing Accuracy for Females: {non_earing_accuracy_female:.2f}%")


    print(f"\n[Ignoring earing]")
    print(f"Male Accuracy (overall): {male_acc_overall:.2f}%")
    print(f"Female Accuracy (overall): {female_acc_overall:.2f}%")

    print(f"\n[Ignoring gender]")
    print(f"earing Accuracy (overall): {earing_acc_overall:.2f}%")
    print(f"Non-earing Accuracy (overall): {non_earing_acc_overall:.2f}%")


    if abs(earing_accuracy_female - earing_accuracy_male) > 10:
        print("\nPotential Bias Detected in earing Prediction")
    else:
        print("\nNo Significant Bias Detected in earing Prediction")

# Test the bias in predictions
evaluate_bias(model, test_loader)
