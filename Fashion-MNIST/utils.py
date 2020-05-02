import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

# test accuracy
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
    print(f'test_accuracy: {test_acc:.4f}')

# Confusion matrix
# code taken from lazyprogrammer
"""
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(cm, classes, normalize = False, title='Confusion matrix', cmap = plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Modified for saving
    plt.savefig('plots/confusion_matrix_f_MNIST.png')

# prep data for matrix
def make_matrix(test_data, test_loader, model, device):

    y_test = test_data.targets.numpy()
    p_test = np.array([])
    
    for inputs, targets in test_loader:
        
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)
  
        p_test = np.concatenate((p_test, predictions.cpu().numpy()))

    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))