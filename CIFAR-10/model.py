import torch.nn as nn
import torch.nn.functional as F

# Model
class CNN(nn.Module):
  def __init__(self, output_dim):
    
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2)
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2)
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2)

    self.fc1 = nn.Linear(128 * 3 * 3, 1024)
    self.fc2 = nn.Linear(1024, output_dim)

  def forward(self, x):
    
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))
    out = out.view(-1, 128 * 3 * 3)
    out = F.dropout(out, p = 0.5)
    out = F.relu(self.fc1(out))
    out = F.dropout(out, p = 0.2)
    out = self.fc2(out)

    return out