import copy
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from models.ViT import ViT_16_mod 
from torchvision import transforms as T

MODELS = {
    #'densenet' : models.densenet121,
    #'efficientnet':models.efficientnet_b0,
    #'googlenet':models.googlenet,
    #'mnasnet':models.mnasnet1_0,
    #'mobilenet':models.mobilenet_v2,
    #'resnet18':models.resnet18,
    #'resnet50':models.resnet50,
    #'shufflenet':models.shufflenet_v2_x1_0,
    'ViT': ViT_16_mod,
}

DATASETS = {
    'CIFAR10' : datasets.CIFAR10,
    #'STL10' : datasets.STL10,
    #'SVHN' : datasets.SVHN,
    #'CIFAR100' : datasets.CIFAR100,
    #'Caltech101': datasets.Caltech101,
    #'DTD': datasets.DTD,
    #'FGVCAircraft': datasets.FGVCAircraft,
    #'Flowers102': datasets.Flowers102,
    #'OxfordPets': datasets.OxfordIIITPet,
    #'TinyImageNet': datasets.ImageFolder  

}

def embedder1(model):
    return nn.Sequential(*list(model.children())[:-1])
def embedder2(model):
    return nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))

def vit_input_transforms(dataset_name: str):
    # Match the 224 pipeline you use in training_original.py for ViT
    mean = {
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4867, 0.4408),
        'TinyImageNet': (0.485, 0.456, 0.406),
    }
    std = {
        'CIFAR10': (0.2023, 0.1994, 0.2010),
        'CIFAR100': (0.2675, 0.2565, 0.2761),
        'TinyImageNet': (0.229, 0.224, 0.225),
    }
    # Train-time ViT path in your code resizes to 224 and normalizes accordingly:contentReference[oaicite:5]{index=5}
    common = [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean[dataset_name], std=std[dataset_name]),
    ]
    # For embeddings, we don’t need heavy aug; keep it deterministic
    return T.Compose(common), T.Compose(common)




class CustomDatasetLoader:

    def __init__(self,
                 dataset : str = "",
                 root : str = "",
                 download : bool = False):


        #shuffling = True might be required for multiple runs

        self.dataset_name = dataset

        # Define dataset-specific transforms
        mean = {
            'CIFAR10': (0.4914, 0.4822, 0.4465),
            'CIFAR100': (0.5071, 0.4867, 0.4408),
            'TinyImageNet': (0.485, 0.456, 0.406),
        }

        std = {
            'CIFAR10': (0.2023, 0.1994, 0.2010),
            'CIFAR100': (0.2675, 0.2565, 0.2761),
            'TinyImageNet': (0.229, 0.224, 0.225),
        }

        transform_train = {
            'CIFAR10': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean['CIFAR10'], std=std['CIFAR10'])
            ]),
            'CIFAR100': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean['CIFAR100'], std=std['CIFAR100'])
            ]),
            'TinyImageNet': transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean['TinyImageNet'], std=std['TinyImageNet'])
            ])
        }

        transform_test = {
            'CIFAR10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean['CIFAR10'], std=std['CIFAR10'])
            ]),
            'CIFAR100': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean['CIFAR100'], std=std['CIFAR100'])
            ]),
            'TinyImageNet': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean['TinyImageNet'], std=std['TinyImageNet'])
            ])
        }

        transform_val = {
            'TinyImageNet': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean['TinyImageNet'], std=std['TinyImageNet'])
            ])
        }
        
        # Select appropriate transformations
        train_transform = transform_train.get(dataset, transform_test.get(dataset, None))
        test_transform = transform_test.get(dataset, train_transform)
        val_transform = transform_val.get(dataset, train_transform)

        if train_transform is None:
            raise ValueError(f"No transformation found for dataset {dataset}")

        if dataset == "TinyImageNet":
            train_dir = os.path.join(root, "train")
            val_dir = os.path.join(root, "val")
            test_dir = os.path.join(root, "test")

            self.train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
            self.val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
            self.test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

            self.dataset = ConcatDataset([self.train_dataset, self.val_dataset])

        elif dataset in ["CIFAR10", "CIFAR100"]:
            train_params = {'root': root, 'train': True, 'transform': train_transform, 'download': download}
            test_params = {'root': root, 'train': False, 'transform': test_transform, 'download': download}


            self.train_dataset = DATASETS[dataset](**train_params)
            self.test_dataset = DATASETS[dataset](**test_params)
            self.dataset = ConcatDataset([self.train_dataset, self.test_dataset])


        # DataLoader over the entire dataset (or concatenated dataset)
        self.loader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        self.size = len(self.dataset)


def get_model(model_name: str, dataset_name: str, num_classes: int, checkpoint_path: Optional[str] = None) -> nn.Module:
    
    if model_name == 'ViT':
        model = ViT_16_mod(n_classes=num_classes)
        print(f"Looking for checkpoint at: {checkpoint_path}")
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded ViT checkpoint from {checkpoint_path}")
        elif checkpoint_path:
            print(f"Warning: Checkpoint path {checkpoint_path} not found. Using randomly initialized ViT.")
        return model    
    
        
    if model_name not in MODELS:
        raise ValueError(f"{model_name} not known.")

    model = MODELS[model_name](pretrained=False)  # No pretraining since we're using a custom checkpoint
    
    # Modify ResNet architecture for CIFAR100
    if model_name in ['resnet18', 'resnet50'] and dataset_name in ['CIFAR100', 'TinyImageNet']:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
        model.maxpool = nn.Identity()
        
        # Ensure fc exists before accessing in_features
        if hasattr(model, 'fc') and model.fc is not None:
            in_features = model.fc.in_features
        else:
            raise ValueError(f"Error: Model {model_name} does not have a valid 'fc' layer before modification.")

        # Match original architecture (Dropout + Linear)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        ).to('cuda')

    elif model_name in ['resnet18', 'resnet50', 'googlenet', 'shufflenet']:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)  # Default modification for other datasets

    print(f"Looking for checkpoint at: {checkpoint_path}")

    # Load checkpoint after modifying the classifier
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Checkpoint file exists, proceeding to load.")

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)  # strict=False to allow partial loading
            print(f"Loaded checkpoint (state_dict) from {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint directly from {checkpoint_path}")
    elif checkpoint_path:
        print(f"Warning: Checkpoint path {checkpoint_path} not found. Using randomly initialized weights.")

    return model





def save_embeddings_to_npz(embedding,
                           labels,
                           DIR : str,
                           folder : str,
                           sub_folder : str,
                           n_model : str,
                           file_name : str):
    if not os.path.isdir(f"{DIR}/{folder}"):
        os.mkdir(f"{DIR}/{folder}")
    if not os.path.isdir(f"{DIR}/{folder}/{sub_folder}"):
        os.mkdir(f"{DIR}/{folder}/{sub_folder}")

    np.savez_compressed(f"{DIR}/{folder}/{sub_folder}/{file_name}_m{n_model}", embeddings=embedding, labels=labels)

class CustomBackboneModel:

    DATASET_NUM_CLASSES = {
        "CIFAR10": 10,
        "STL10": 10,
        "SVHN": 10,
        "CIFAR100": 100,
        "Caltech101": 101,
        "DTD": 47,
        "Flowers102": 102,
        "FGVCAircraft": 100,
        "OxfordPets": 37,
        "TinyImageNet": 200,
    }

    def __init__(self, model: str, dataset: str, checkpoint_path: Optional[str] = None):

        self.model_name = model
        self.dataset_name = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if self.dataset_name in self.DATASET_NUM_CLASSES:
            self.num_classes = self.DATASET_NUM_CLASSES[self.dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.model = get_model(self.model_name, self.dataset_name, self.num_classes, checkpoint_path)

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def to_num_classes(self):
        if self.model_name == 'densenet':
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, self.num_classes)
        elif self.model_name in ['efficientnet', 'mnasnet', 'mobilenet']:
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[1].in_features  # Extract from Sequential
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features, self.num_classes)
                )
            else:
                raise AttributeError(f"{self.model_name} classifier is not in the expected Sequential format.")
        elif self.model_name in ['googlenet', 'resnet18', 'resnet50', 'shufflenet']:
            # Fix for Sequential issue
            if isinstance(self.model.fc, nn.Sequential):
                in_features = self.model.fc[-1].in_features  # Extract from last layer in Sequential
            else:
                in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"{self.model_name} has a custom last layer name.")
        return self


    @property
    def last_layer(self):
        if self.model_name in ['densenet', 'efficientnet', 'mnasnet', 'mobilenet']:
            return self.model.classifier
        elif self.model_name in ['googlenet', 'resnet18', 'resnet50', 'shufflenet']:
            return self.model.fc
        else:
            raise ValueError(f"{self.model_name} has a custom last layer name.")

    @property
    def embedder(self):
        _embedder = embedder1
        if self.model_name in ['densenet', 'mnasnet', 'mobilenet', 'shufflenet']:
            _embedder = embedder2

        return _embedder(self.model)

        if self.model_name == 'ViT':
            return None

        raise ValueError(f"Unknown model for embedder: {self.model_name}")

    def _embed_loader(self, dataset: CustomDatasetLoader):
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for _, (inputs, label) in enumerate(tqdm(dataset.loader, desc=f"embedding {dataset.dataset_name} by {self.model_name}", leave=False)):
                inputs = inputs.to(self.device)

                if self.model_name == 'ViT':
                    # Grab pre-FC features via forward_encoder (CLS, 768d):contentReference[oaicite:6]{index=6}
                    embedding = self.model.forward_encoder(inputs)
                else:
                    embedding = self.embedder(inputs)
                    
                embeddings.append(embedding.cpu().numpy())
                labels.append(label)

        return np.concatenate(embeddings), np.concatenate(labels)

    def embed_dataset(self, dataset: CustomDatasetLoader):
        # Split the dataset into train and test
        train_loader = DataLoader(dataset.train_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(dataset.test_dataset, batch_size=64, shuffle=False)
        
        # Check if validation set exists
        has_val = hasattr(dataset, 'val_dataset')

        if has_val:
            val_loader = DataLoader(dataset.val_dataset, batch_size=64, shuffle=False)
            
        # Embed training dataset
        dataset.loader = train_loader
        train_embeddings, train_labels = self._embed_loader(dataset)

        # Embed testing dataset
        dataset.loader = test_loader
        test_embeddings, test_labels = self._embed_loader(dataset)

        # Embed validation dataset if available
        val_embeddings, val_labels = None, None
        if has_val:
            dataset.loader = val_loader
            val_embeddings, val_labels = self._embed_loader(dataset)

        # Embed the entire dataset (train + test)
        full_loader = DataLoader(dataset.dataset, batch_size=64, shuffle=False)
        dataset.loader = full_loader
        full_embeddings, full_labels = self._embed_loader(dataset)

        # Reshape embeddings to 2D (flatten the embeddings)
        train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
        test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)
        
        if has_val:
            val_embeddings = val_embeddings.reshape(val_embeddings.shape[0], -1)

        full_embeddings = full_embeddings.reshape(full_embeddings.shape[0], -1)

        # Return values based on dataset type
        if has_val:
            return train_embeddings, train_labels, test_embeddings, test_labels, val_embeddings, val_labels, full_embeddings, full_labels
        else:
            return train_embeddings, train_labels, test_embeddings, test_labels, full_embeddings, full_labels
    
    
    






