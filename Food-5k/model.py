from torchvision import models
import torch.nn as nn

# Get the VGG model and modify
def get_model():
    
    # Get VGG model
    model = models.vgg16(pretrained=True)

    # Freeze VGG weights
    for param in model.parameters():
        param.requires_grad = False

    n_features = model.classifier[0].in_features
    
    # Modify the model
    model.classifier = nn.Linear(n_features, 2)

    return model
