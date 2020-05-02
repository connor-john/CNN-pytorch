import torch
import torch.nn as nn

# CNN model
class CNN(nn.Module):
  def __init__(self, output_dim):
    
    super(CNN, self).__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 2),
        nn.ReLU(),
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64, out_channels = 128,  kernel_size = 3, stride = 2),
        nn.ReLU())
    
    self.dense = nn.Sequential(
        nn.Dropout(0.2),
        # convolutional arithmetic
        nn.Linear(128 * 2 * 2, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, output_dim))
  
  def forward(self, x):
    
    out = self.conv(x)
    out = out.view(out.size(0), -1)
    out = self.dense(out)

    return out