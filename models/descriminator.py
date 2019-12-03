import torch
import torch.nn as nn
from torchvision.models import resnet34


class Descriminator(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(512, out_channels)
        self.model = model

    def forward(self, x):
        return self.model.forward(x).sigmoid()
