import torchvision
import torchvision.transforms as transforms
import numpy as np

# Get data
def get_data():

    train_data = torchvision.datasets.FashionMNIST(
        root = '.', 
        train = True, 
        transform = transforms.ToTensor(),
        download = True)
    
    test_data = torchvision.datasets.FashionMNIST(
        root = '.', 
        train = False, 
        transform = transforms.ToTensor(),
        download = True)

    labels = len(set(train_data.targets.numpy()))

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

# make environment
def make_env(batch_size):
    
    train_data, test_data, labels = get_data()

    train_loader, test_loader = prep_data(train_data, test_data, batch_size)

    return train_loader, test_loader, labels
