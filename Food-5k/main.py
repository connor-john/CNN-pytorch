import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

from model import get_model
from preprocessing import make_env

# Hyper params -- for pref
batch_size = 128
n_epochs = 5

# Training loop
# Using batch gradient descent
def train(model, criterion, optimizer, train_loader, test_loader, n_epochs, device):
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    for i in range(n_epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
                
            # optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[i] = train_loss
        test_losses[i] = test_loss
        
        dt = datetime.now() - t0
        print(f'epoch {i+1}/{n_epochs} | train_Loss: {train_loss:.4f} | test_Loss: {test_loss:.4f} | duration: {dt}')
    
    return train_losses, test_losses

def get_test_acc(test_loader, model, device):
    n_correct = 0.
    n_total = 0.
    for inputs, targets in test_loader:

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)
        
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total
    print(f"test_acc: {test_acc:.4f}")

# main
if __name__ == '__main__':

    # initialise
    train_loader, test_loader, test_data, train_loader_fixed = make_env(batch_size)
    model = get_model()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train
    train_losses, test_losses = train(model, criterion, optimizer, train_loader, test_loader, n_epochs = n_epochs, device = device)

    # accuracy
    get_test_acc(test_loader, model, device)


