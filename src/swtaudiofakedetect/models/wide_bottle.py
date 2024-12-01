from typing import Optional

import torch
import torch.nn as nn

from swtaudiofakedetect.dataset_transform import Transformer
from swtaudiofakedetect.residual_block import ResNetBasicBlock, ResNetBottleneckBlock


class Wide19Bottle(nn.Module):  # type: ignore
    """
    20 layer network consisting of 19 2d convolutional layers and 1 fully connected layer.
    The 19 2d convolutional layers are grouped into 8 residual blocks inspired by the original ResNet implementation.
    Total number of parameters: ~112K.
    Expects input shape (1, swt_levels+1, sample_count) = (1, 15, 16384).
    """

    def __init__(self, gpu_transform: Optional[Transformer] = None) -> None:
        super().__init__()
        self.transform = gpu_transform

        # input: (B, 1, 15, 16384), output: (B, 8, 8, 2048), 7 conv layers
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(
                1, 4, kernel_size=(3, 4), stride=(1, 2), padding=(1, 2), dilation=(1, 2), activation=nn.LeakyReLU()
            ),
            ResNetBasicBlock(4, 8, kernel_size=3, stride=(1, 2), activation=nn.LeakyReLU()),
            ResNetBottleneckBlock(8, kernel_size=3, bottleneck=2, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2, padding=(1, 0)),
        )

        # input: (B, 8, 8, 2048), output: (B, 32, 4, 512), 7 conv layers
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(
                8, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 2), dilation=(1, 2), activation=nn.LeakyReLU()
            ),
            ResNetBasicBlock(16, 32, kernel_size=3, stride=(1, 2), activation=nn.LeakyReLU()),
            ResNetBottleneckBlock(32, kernel_size=3, bottleneck=2, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
        )

        # input: (B, 32, 4, 512), output: (B, 128, 1, 64), 5 conv layers
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(32, 64, kernel_size=3, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
            ResNetBottleneckBlock(64, kernel_size=3, bottleneck=4, expansion=2, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2, stride=(1, 2)),
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 1 * 64, 2))

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


class Wide32Bottle(nn.Module):  # type: ignore
    """
    33 layer network consisting of 32 2d convolutional layers and 1 fully connected layer.
    The 34 2d convolutional layers are grouped into 14 residual blocks inspired by the original ResNet implementation.
    Total number of parameters: ~639K.
    Expects input shape (1, swt_levels+1, sample_count) = (1, 15, 16384).
    """

    def __init__(self, gpu_transform: Optional[Transformer] = None):
        super().__init__()
        self.transform = gpu_transform

        # input: (B, 1, 15, 16384), output: (B, 8, 15, 4096), 6 conv layers
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(
                1, 4, kernel_size=(3, 4), stride=(1, 2), padding=(1, 2), dilation=(1, 2), activation=nn.LeakyReLU()
            ),
            ResNetBasicBlock(4, 8, kernel_size=3, stride=(1, 2), activation=nn.LeakyReLU()),
            ResNetBasicBlock(8, 8, kernel_size=3, activation=nn.LeakyReLU()),
        )

        # input: (B, 8, 15, 4096), output: (B, 32, 15, 1024), 6 conv layers
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(
                8, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 2), dilation=(1, 2), activation=nn.LeakyReLU()
            ),
            ResNetBasicBlock(16, 32, kernel_size=3, stride=(1, 2), activation=nn.LeakyReLU()),
            ResNetBasicBlock(32, 32, kernel_size=3, activation=nn.LeakyReLU()),
        )  # 12 Conv blocks until here

        # input: (B, 32, 15, 1024), output: (B, 64, 4, 256), 10 conv layers
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(32, 48, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBasicBlock(48, 64, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBottleneckBlock(64, bottleneck=2, kernel_size=3, activation=nn.LeakyReLU()),
            ResNetBottleneckBlock(64, bottleneck=2, kernel_size=3, activation=nn.LeakyReLU()),
        )

        # input: (B, 64, 4, 256), output: (B, 128, 1, 64), 10 conv layers
        self.layer4 = nn.Sequential(
            ResNetBasicBlock(64, 96, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBasicBlock(96, 128, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBottleneckBlock(128, bottleneck=4, kernel_size=3, activation=nn.LeakyReLU()),
            ResNetBottleneckBlock(128, bottleneck=4, kernel_size=3, activation=nn.LeakyReLU()),
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 1 * 64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass"""
        if self.transform is not None:
            x = self.transform(x)

        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc(out)
        return out

    def count_parameters(self) -> int:
        """Count the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters())
