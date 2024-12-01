from typing import Optional, Tuple

import torch
import torch.nn as nn

from swtaudiofakedetect.dataset_transform import Transformer
from swtaudiofakedetect.utils import get_conv2d_output_shape, get_maxpool2d_output_shape


class DCNN(torch.nn.Module):  # type: ignore
    """Deep CNN with dilated convolutions."""

    def __init__(self, x_size: int, y_size: int, gpu_transform: Optional[Transformer] = None) -> None:
        super(DCNN, self).__init__()
        self.transform = gpu_transform

        # x_size: size of time dimension
        # y_size: size of packets dimension

        # expects shape [batch, channels, time, packets]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, 1, 1, padding=0),
            nn.PReLU(),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 96, 3, 1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(96, affine=False),
            nn.Conv2d(96, 128, 3, 1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128, affine=False),
            nn.Conv2d(128, 32, 3, 1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.6),
        )

        time_size, packets_size = self.calc_size_after_cnn(x_size, y_size)

        # expects shape [batch, time, channels, packets]
        self.dil_conv = nn.Sequential(
            nn.BatchNorm2d(time_size, affine=True),
            nn.Conv2d(time_size, time_size, 3, 1, padding=1, dilation=1),
            nn.PReLU(),
            nn.BatchNorm2d(time_size, affine=True),
            nn.Conv2d(time_size, time_size, 5, 1, padding=2, dilation=2),
            nn.PReLU(),
            nn.BatchNorm2d(time_size, affine=True),
            nn.Conv2d(time_size, time_size, 7, 1, padding=2, dilation=4),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        channels_size, packets_size = self.calc_size_after_dil(64, packets_size)

        self.fc = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(channels_size * packets_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.transform is not None:
            x = self.transform(x)

        # x has shape [batch, channels, packets, time]

        x = self.cnn(x.permute(0, 1, 3, 2))  # permute to [batch, channels, time, packets]
        x = x.permute(0, 2, 1, 3).contiguous()  # permute to [batch, time, channels, packets]
        x = self.dil_conv(x)
        x = self.fc(x).mean(1)

        return x

    def count_parameters(self) -> int:
        """Count the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def calc_size_after_cnn(x: int, y: int) -> Tuple[int, int]:
        x, y = get_conv2d_output_shape((x, y), 3, padding=2)
        x, y = get_maxpool2d_output_shape((x, y), 2)
        x, y = get_conv2d_output_shape((x, y), 1, padding=0)
        x, y = get_conv2d_output_shape((x, y), 3, padding=1)
        x, y = get_maxpool2d_output_shape((x, y), 2)
        x, y = get_conv2d_output_shape((x, y), 3, padding=1)
        x, y = get_conv2d_output_shape((x, y), 3, padding=1)
        x, y = get_conv2d_output_shape((x, y), 3, padding=1)
        x, y = get_maxpool2d_output_shape((x, y), 2)
        return x, y

    @staticmethod
    def calc_size_after_dil(x: int, y: int) -> Tuple[int, int]:
        x, y = get_conv2d_output_shape((x, y), 3, 1, 1, 1)
        x, y = get_conv2d_output_shape((x, y), 5, 1, 2, 2)
        x, y = get_conv2d_output_shape((x, y), 7, 1, 2, 4)
        return x, y
