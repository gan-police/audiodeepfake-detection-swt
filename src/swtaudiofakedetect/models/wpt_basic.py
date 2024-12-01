from typing import Optional

import torch
import torch.nn as nn

from swtaudiofakedetect.dataset_transform import Transformer
from swtaudiofakedetect.residual_block import ResNetBasicBlock


class WptBasic(nn.Module):  # type: ignore
    """
    13 layer network consisting of 12 2d convolutional layers and 1 fully connected layer.
    Total number of parameters: ~495K.
    Expects input shape (1, y_size, x_size).
    """

    def __init__(self, x_size: int, y_size: int, gpu_transform: Optional[Transformer] = None) -> None:
        super().__init__()
        self.transform = gpu_transform

        # input: (B, 1, Y, X), output: (B, 16, Y/2, X/2), 4 conv layers
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(1, 4, kernel_size=5, activation=nn.LeakyReLU()),
            ResNetBasicBlock(4, 8, kernel_size=3, expansion=2, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
        )

        # input: (B, 16, Y/2, X/2), output: (B, 64, Y/4, X/4), 4 conv layers
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(16, 24, kernel_size=3, activation=nn.LeakyReLU()),
            ResNetBasicBlock(24, 32, kernel_size=3, expansion=2, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
        )

        # input: (B, 64, Y/4, X/4), output: (B, 128, Y/16, X/16), 4 conv layers
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(64, 96, kernel_size=3, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
            ResNetBasicBlock(96, 128, kernel_size=3, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
        )

        x_out_size = x_size // 16
        y_out_size = y_size // 16
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * x_out_size * y_out_size, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass"""
        if self.transform is not None:
            x = self.transform(x)

        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)
        return out

    def count_parameters(self) -> int:
        """Count the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters())
