import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Get data
def get_data():

    # Augment data
    transformer_train = torchvision.transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),                                            
    ])

    # Get Data -- Using Fashion MNIST dataset
    train_data = torchvision.datasets.CIFAR10(
        root = '.', 
        train = True, 
        transform = transformer_train,
        download = True) 
    test_data = torchvision.datasets.CIFAR10(
        root = '.', 
        train = False, 
        transform = transforms.ToTensor(),
        download = True)

    labels = len(set(train_data.targets))

    return train_data, test_data, labels
    
# prep data
def prep_data(train, test, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset = train,
                                           batch_size = batch_size,
                                           shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test,
                                           batch_size = batch_size,
                                           shuffle = False)
    
    return train_loader, test_loader

# Batch normalisation
def batch_norm():
    return None

# make environment
def make_env(batch_size):
    
    train_data, test_data, labels = get_data()

    train_loader, test_loader = prep_data(train_data, test_data, batch_size)

    print('environment made...')
    return train_loader, test_loader, labels, test_data
