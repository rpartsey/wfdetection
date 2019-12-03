import segmentation_models_pytorch as smp
import torch.nn as nn


class SmpUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()
        assert in_channels == 3, "Input should be 3"
        assert out_channels == 1, "Out should be 1"
        self.model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

    def forward(self, x):
        return self.model(x)
