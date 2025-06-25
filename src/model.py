import torch.nn as nn
from torchvision import models

def get_resnet18_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Grayscale
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification
    return model

