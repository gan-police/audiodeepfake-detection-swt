from typing import Optional

import torch
import torch.nn as nn

from swtaudiofakedetect.dataset_transform import Transformer


class Wide6l2dCNN(nn.Module):  # type: ignore
    """
    7 layer network consisting of 6 2d convolutional layers and a final fully connected layer.
    Total number of parameters: ~164k.
    Expects input shape (1, swt_levels+1, sample_count) = (1, 15, 16384).
    """

    def __init__(self, activation: nn.Module = nn.ReLU(), gpu_transform: Optional[Transformer] = None):
        super().__init__()
        self.transform = gpu_transform

        conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=(1, 2), padding=(1, 2), dilation=(1, 2))
        # input: (B, 1, 15, 16384), output: (B, 4, 15, 8192)
        self.layer1 = nn.Sequential(conv1, nn.BatchNorm2d(4), activation)

        conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=(1, 2), padding=(1, 2), dilation=(1, 2))
        # input: (B, 4, 15, 8192), output: (B, 8, 15, 4096)
        self.layer2 = nn.Sequential(conv2, nn.BatchNorm2d(8), activation)

        conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=(1, 2), dilation=(1, 2))
        # input: (B, 8, 15, 4096), output: (B, 16, 8, 2048)
        self.layer3 = nn.Sequential(conv3, nn.BatchNorm2d(16), activation)

        conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        # input: (B, 16, 8, 2048), output: (B, 32, 4, 1024)
        self.layer4 = nn.Sequential(conv4, nn.BatchNorm2d(32), activation)

        conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        # input: (B, 32, 4, 1024), output: (B, 64, 2, 512)
        self.layer5 = nn.Sequential(conv5, nn.BatchNorm2d(64), activation)

        conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        # input: (B, 64, 2, 512), output: (B, 128, 1, 256)
        self.layer6 = nn.Sequential(conv6, nn.BatchNorm2d(128), activation)

        # input: (B, 128, 256), output: (B, 2)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 256, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass"""
        if self.transform is not None:
            x = self.transform(x)

        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        return out

    def count_parameters(self) -> int:
        """Count the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters())
