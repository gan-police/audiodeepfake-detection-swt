from typing import Optional

import torch
import torch.nn as nn

from swtaudiofakedetect.dataset_transform import Transformer
from swtaudiofakedetect.residual_block import ResNetBasicBlock


class Wide16Basic(nn.Module):  # type: ignore
    """
    17 layer network consisting of 16 2d convolutional layers and 1 fully connected layer.
    The 16 2d convolutional layers are grouped into 8 residual blocks inspired by the original ResNet implementation.
    Total number of parameters: ~344K.
    Expects input shape (1, swt_levels+1, sample_count) = (1, 15, 16384).
    """

    def __init__(self, gpu_transform: Optional[Transformer] = None):
        super().__init__()
        self.transform = gpu_transform

        # input: (B, 1, 15, 16384), output: (B, 8, 8, 2048), 6 conv layers
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(
                1, 4, kernel_size=(3, 4), stride=(1, 2), padding=(1, 2), dilation=(1, 2), activation=nn.LeakyReLU()
            ),
            ResNetBasicBlock(4, 8, kernel_size=3, stride=(1, 2), activation=nn.LeakyReLU()),
            ResNetBasicBlock(8, 8, kernel_size=3, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2, padding=(1, 0)),
        )

        # input: (B, 8, 8, 2048), output: (B, 32, 4, 512), 6 conv layers
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(
                8, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 2), dilation=(1, 2), activation=nn.LeakyReLU()
            ),
            ResNetBasicBlock(16, 32, kernel_size=3, stride=(1, 2), activation=nn.LeakyReLU()),
            ResNetBasicBlock(32, 32, kernel_size=3, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
        )

        # input: (B, 32, 4, 512), output: (B, 128, 1, 64), 4 conv layers
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(32, 64, kernel_size=3, activation=nn.LeakyReLU()),
            nn.MaxPool2d(2),
            ResNetBasicBlock(64, 128, kernel_size=3, activation=nn.LeakyReLU()),
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


class Wide24Basic(nn.Module):  # type: ignore
    """
    25 layer network consisting of 24 2d convolutional layers and 1 fully connected layer.
    The 24 2d convolutional layers are grouped into 12 residual blocks inspired by the original ResNet implementation.
    Total number of parameters: ~946K.
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
        )

        # input: (B, 32, 15, 1024), output: (B, 64, 4, 256), 6 conv layers
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(32, 48, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBasicBlock(48, 64, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBasicBlock(64, 64, kernel_size=3, activation=nn.LeakyReLU()),
        )

        # input: (B, 64, 4, 256), output: (B, 128, 1, 64), 6 conv layers
        self.layer4 = nn.Sequential(
            ResNetBasicBlock(64, 96, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBasicBlock(96, 128, kernel_size=3, stride=2, activation=nn.LeakyReLU()),
            ResNetBasicBlock(128, 128, kernel_size=3, activation=nn.LeakyReLU()),
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
