import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import sys, os

# Get data
def get_data():

    # Data augmentation
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.CenterCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get Data -- https://mmspg.epfl.ch/downloads/food-image-datasets/
    # Directories for the data in Keras-style
    train_data = datasets.ImageFolder(
        'data/train',
        transform=train_transform
    )

    test_data = datasets.ImageFolder(
        'data/test',
        transform=test_transform
    )

    return train_data, test_data
    
# prep data
def get_loaders(train, test, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset = train,
                                           batch_size = batch_size,
                                           shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test,
                                           batch_size = batch_size,
                                           shuffle = False)
    
    return train_loader, test_loader

# make environment
def make_env(batch_size):
    
    train_data, test_data = get_data()

    train_loader, test_loader = get_loaders(train_data, test_data, batch_size)

    print('environment made...')
    return train_loader, test_loader, test_data
