import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

from model import CNN
from preprocessing import make_env
from utils import get_test_acc, make_matrix

# Hyper params -- for pref
batch_size = 128
n_epochs = 15

# Training loop
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
        
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

    # Get train loss and test loss
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
    print(f'epoch {i+1}/{n_epochs} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | duration: {dt}')
  
  return train_losses, test_losses

# main
if __name__ == '__main__':

    # initialise
    train_loader, test_loader, labels, test_data = make_env(batch_size)
    model = CNN(output_dim = labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train
    train_losses, test_losses = train(model, criterion, optimizer, train_loader, test_loader, n_epochs = n_epochs, device = device)

    # accuracy
    get_test_acc(test_loader, model, device)

    # plot
    make_matrix(test_data, test_loader, model, device)

