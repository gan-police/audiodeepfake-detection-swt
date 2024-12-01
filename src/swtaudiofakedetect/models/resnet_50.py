from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from swtaudiofakedetect.dataset_transform import Transformer


class WaveResNet50(nn.Module):  # type: ignore
    """
    ResNet-50 model with the final fully connected layer adjusted to two output channels.
    https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

    Total number of parameters: ~23.51M.
    Expects input of shape (3, 224, 224) or (3, 225, 225).
    """

    def __init__(self, gpu_transform: Optional[Transformer] = None, pretrained: bool = False):
        super().__init__()
        self.transform = gpu_transform

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None, progress=False)
        self.resnet.fc = nn.Linear(2048, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass"""
        if self.transform is not None:
            x = self.transform(x)

        return self.resnet(x)

    def count_parameters(self) -> int:
        """Count the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters())
