from typing import Optional

import torch
import torch.nn as nn

from swtaudiofakedetect.dataset_transform import Transformer


class Wide6l1dCNN(nn.Module):  # type: ignore
    """
    7 layer network consisting of 6 1d convolutional layers and a final fully connected layer.
    Total number of parameters: ~139k.
    Expects input shape (swt_levels+1, sample_count) = (15, 16384).
    """

    def __init__(self, activation: nn.Module = nn.ReLU(), gpu_transform: Optional[Transformer] = None) -> None:
        super().__init__()
        self.transform = gpu_transform

        conv1 = nn.Conv1d(15, 24, kernel_size=3, stride=2, padding=2, dilation=2)
        # input: (B, 15, 16384), output: (B, 24, 8192)
        self.layer1 = nn.Sequential(conv1, nn.BatchNorm1d(24), activation)

        conv2 = nn.Conv1d(24, 32, kernel_size=3, stride=2, padding=2, dilation=2)
        # input: (B, 24, 8192), output: (B, 32, 4096)
        self.layer2 = nn.Sequential(conv2, nn.BatchNorm1d(32), activation)

        conv3 = nn.Conv1d(32, 48, kernel_size=3, stride=2, padding=2, dilation=2)
        # input: (B, 32, 4096), output: (B, 48, 2048)
        self.layer3 = nn.Sequential(conv3, nn.BatchNorm1d(48), activation)

        conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        # input: (B, 48, 2048), output: (B, 64, 1024)
        self.layer4 = nn.Sequential(conv4, nn.BatchNorm1d(64), activation)

        conv5 = nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1, dilation=1)
        # input: (B, 64, 1024), output: (B, 96, 512)
        self.layer5 = nn.Sequential(conv5, nn.BatchNorm1d(96), activation)

        conv6 = nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        # input: (B, 96, 512), output: (B, 128, 256)
        self.layer6 = nn.Sequential(conv6, nn.BatchNorm1d(128), activation)

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

    def count_parameters(self) -> None:
        """Count the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters())
