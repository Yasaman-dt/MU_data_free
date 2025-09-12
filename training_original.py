# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
from models.allcnn import AllCNN
from models.ViT import ViT_16_mod
from models.swin_transformer import swin_tiny_patch4_window7_224
from opts import OPT as opt
import os 
import wandb
run = wandb.init()


mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'TinyImageNet': (0.485, 0.456, 0.406),
        }

std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'TinyImageNet': (0.229, 0.224, 0.225),
        }


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
])

transform_train_tiny = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])

transform_test_tiny = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])


if opt.model in ('ViT', 'swint'):
    transform_train = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset]),
    ])


def trainer(removed=None):
    # Initialize the model
    if opt.model == 'resnet18':
        model= torchvision.models.resnet18(pretrained=True).to('cuda')
    elif opt.model=='resnet34':
        model= torchvision.models.resnet34(pretrained=True).to('cuda')
    elif opt.model=='resnet50':
        model = torchvision.models.resnet50(pretrained=True).to('cuda')
    elif opt.model=='AllCNN':
        model = AllCNN(n_channels=3, num_classes=opt.num_classes).to('cuda')
    elif opt.model == 'ViT':           
        model = ViT_16_mod(n_classes=opt.num_classes).to('cuda')
    elif opt.model == "swint":
        model = swin_tiny_patch4_window7_224(pretrained=False, num_classes=opt.num_classes).to('cuda')   
    else:
        raise ValueError(f"Unknown model: {opt.model}")


    if opt.dataset == 'cifar10':
        os.makedirs(f'./weights/chks_cifar10/original', exist_ok=True)
        # Load CIFAR-10 data
        trainset = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_test)
        
        if 'resnet' in opt.model:    
            model.fc = nn.Linear(model.fc.in_features, opt.num_classes).to('cuda')

    elif opt.dataset == 'cifar100':
        os.makedirs(f'./weights/chks_cifar100/original', exist_ok=True)
        trainset = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_test)
        
        if 'resnet' in opt.model:    
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
            model.maxpool = nn.Identity()
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to('cuda')

    elif opt.dataset == 'TinyImageNet':
        #dataloader
        os.makedirs(f'./weights/chks_TinyImageNet/original', exist_ok=True)
        
        
        if opt.model == 'ViT':
            # use the 224×224 ViT transforms you defined above
            trainset = torchvision.datasets.ImageFolder(opt.data_path+'/TinyImageNet/train', transform=transform_train)
            testset  = torchvision.datasets.ImageFolder(opt.data_path+'/TinyImageNet/val',   transform=transform_test)
            model.heads[-1] = nn.Linear(model.heads[-1].in_features, opt.num_classes).to('cuda')        
        else:
            trainset = torchvision.datasets.ImageFolder(root=opt.data_path+'/TinyImageNet/train',transform=transform_train_tiny)
            #testset = torchvision.datasets.ImageFolder(root=opt.data_path+'/TinyImageNet/val/images',transform=transform_test_tiny)
            testset = torchvision.datasets.ImageFolder(root=opt.data_path+'/TinyImageNet/val', transform=transform_test_tiny)

            if 'resnet' in opt.model:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
                model.maxpool = nn.Identity()
                model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to('cuda')
                

            elif opt.model == 'AllCNN':                     
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, opt.num_classes).to('cuda')            
                
    #dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=opt.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=opt.num_workers)

    epochs=300
    criterion = nn.CrossEntropyLoss(label_smoothing=0.4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-5)
    if opt.dataset == 'TinyImageNet':
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #learning rate decay
    else:
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)


    # Train the network
    best_acc = 0.0
    patience = 30  # Stop training if no improvement for this many epochs
    patience_counter = 0  # Counter to track epochs without improvement
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        correct, total = 0, 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = 100 * correct / total
        train_loss = running_loss / len(trainloader)
        train_scheduler.step()
        torch.save(model.state_dict(), f'weights/chks_{opt.dataset}/original/best_checkpoint_{opt.model}_m{opt.n_model}.pth')

        if epoch % 1 == 0:        
            model.eval()
            correct = 0
            total = 0
            val_running_loss = 0.0  # Initialize validation loss

            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Compute validation loss
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
            val_acc = 100 * correct / total
            val_loss = val_running_loss / len(testloader)  # Compute average validation loss

    
                # Check if this is the best validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0  # Reset patience counter
                torch.save(model.state_dict(), f'weights/chks_{opt.dataset}/original/best_checkpoint_{opt.model}_m{opt.n_model}.pth')
                print(f"New best model saved with Val Acc: {best_acc:.3f}")
            else:
                patience_counter += 1  # No improvement

            print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Best Acc: {best_acc:.3f}')
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})

            # Check for early stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break  # Exit training loop
    
    
    
    return best_acc


if __name__ == '__main__':
    trainer()