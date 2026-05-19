import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os

# Set device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset Setup (CelebA)
data_dir = './data'  # You can specify your desired directory to store the data

# Transformations (Standard CelebA transformations)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(178, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean and std
])

transform_test = transforms.Compose([
    transforms.Resize(178),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean and std
])

# Download and load the CelebA dataset
train_dataset = datasets.CelebA(root=data_dir, split='train', transform=transform_train, download=True)
test_dataset = datasets.CelebA(root=data_dir, split='test', transform=transform_test, download=True)

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
        
        # Earing classification head (binary: Earing vs Not Earing)
        self.earing_head = nn.Linear(512, 1)
        
        # Gender classification head (binary: Male vs Female)
        self.gender_head = nn.Linear(512, 2)
    
    def forward(self, x):
        # Forward pass through ResNet layers
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        # Pass through the fully connected layer
        x = self.fc(x)
        
        # Pass through the Earing head
        earing_output = self.earing_head(x)
        
        # Pass through the Gender head
        gender_output = self.gender_head(x)
        
        return earing_output, gender_output

# Initialize the dual-head model
model = DualHeadResNet18()
model = model.to(device)

# Loss functions
criterion_earing = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss for Earing vs Not Earing
criterion_gender = nn.CrossEntropyLoss()  # CrossEntropyLoss for Gender Classification (Male vs Female)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

earing_idx = 31  # Index for "Earing" label in CelebA
gender_idx = 20


# Initialize the best accuracy to a very low number
best_test_accuracy = 0.0

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss_earing = 0.0
    running_loss_gender = 0.0
    correct_train_earing = 0
    correct_train_gender = 0
    total_train = 0
    
    for inputs, full_labels in train_loader:
        # Extract Earing and Gender labels from the full label vector
        earing_labels = full_labels[:, earing_idx]  # Earing label (index 31)
        gender_labels = full_labels[:, gender_idx]  # Gender label (index 20: 0 for Male, 1 for Female)

        # Ensure both tensors are on the correct device
        inputs, earing_labels, gender_labels = inputs.to(device), earing_labels.to(device), gender_labels.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass through the dual-head model
        earing_output, gender_output = model(inputs)
        
        # Calculate loss for Earing vs Not Earing (binary classification)
        loss_earing = criterion_earing(earing_output.squeeze(), earing_labels.float())
        
        # Calculate loss for Gender Classification (multi-class)
        loss_gender = criterion_gender(gender_output, gender_labels)
        
        # Total loss
        total_loss = loss_earing + loss_gender
        
        # Backward pass
        total_loss.backward()
        optimizer.step()  # Update model parameters
        
        running_loss_earing += loss_earing.item()
        running_loss_gender += loss_gender.item()
        
        # Convert logits to predictions
        predicted_earing = torch.round(torch.sigmoid(earing_output)).squeeze()
        _, predicted_gender = torch.max(gender_output, 1)
        
        # Calculate accuracy for Earing and Gender
        total_train += earing_labels.size(0)
        correct_train_earing += predicted_earing.eq(earing_labels).sum().item()
        correct_train_gender += predicted_gender.eq(gender_labels).sum().item()
    
    # Print statistics every epoch for Training
    print(f"Epoch {epoch+1}/{num_epochs}, Earing Loss: {running_loss_earing/len(train_loader)}, Gender Loss: {running_loss_gender/len(train_loader)}")
    print(f"Train Accuracy - Earing: {100 * correct_train_earing / total_train}%, Gender: {100 * correct_train_gender / total_train}%")
    
    # Evaluate on the test set after each epoch
    model.eval()  # Set the model to evaluation mode
    correct_test_earing = 0
    correct_test_gender = 0
    total_test = 0
    
    with torch.no_grad():  # No gradients required for evaluation
        for inputs, full_labels in test_loader:
            # Extract Earing and Gender labels from the full label vector
            earing_labels = full_labels[:, earing_idx]
            gender_labels = full_labels[:, gender_idx]

            inputs, earing_labels, gender_labels = inputs.to(device), earing_labels.to(device), gender_labels.to(device)
            
            # Forward pass
            earing_output, gender_output = model(inputs)
            
            # Convert logits to predictions
            predicted_earing = torch.round(torch.sigmoid(earing_output)).squeeze()
            _, predicted_gender = torch.max(gender_output, 1)
            
            total_test += earing_labels.size(0)
            correct_test_earing += predicted_earing.eq(earing_labels).sum().item()
            correct_test_gender += predicted_gender.eq(gender_labels).sum().item()
    
    # Print statistics for Test accuracy
    print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy - Earing: {100 * correct_test_earing / total_test}%, Gender: {100 * correct_test_gender / total_test}%")
    
    # Save the model checkpoint if it has the best combined test accuracy so far
    combined_test_accuracy = (correct_test_earing + correct_test_gender) / total_test  # Combined accuracy for both tasks
    
    if combined_test_accuracy > best_test_accuracy:
        best_test_accuracy = combined_test_accuracy
        # Save the best model checkpoint
        torch.save(model.state_dict(), 'best_dual_head_model.pth')
        print(f"Best model checkpoint saved at epoch {epoch+1} with combined test accuracy: {combined_test_accuracy * 100}%")

# Final model saving (if you want to save the last model)
torch.save(model.state_dict(), 'last_dual_head_model.pth')
