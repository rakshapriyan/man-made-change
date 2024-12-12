import torch
import torch.nn as nn
from torchvision import models

# Define a Siamese Network for Change Detection
class ResCNN(nn.Module):
    def __init__(self, base_model):
        super(ResCNN, self).__init__()
        self.feature_extractor = base_model  # Pretrained CNN (e.g., ResNet)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Change or No-Change
        )

    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        diff = torch.abs(feat1 - feat2)  # Compute feature difference
        out = self.fc(diff)
        return out

# Example Model Initialization
base_model = models.resnet18(pretrained=True)
base_model.fc = nn.Identity()  # Remove classification head
model = ResCNN(base_model)


{'event': 'interaction', 'type': 'mousedown', 'coordinates': [28.1889374044219, 76.85657057751139]}
{'event': 'interaction', 'type': 'mousedown', 'coordinates': [28.176643457015388, 76.87137449764242]}
