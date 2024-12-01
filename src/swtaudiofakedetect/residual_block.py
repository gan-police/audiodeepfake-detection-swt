from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module = nn.Identity(),
        shortcut: nn.Module = nn.Identity(),
        activation: nn.Module = nn.ReLU(),
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        self.block, self.shortcut, self.activation = block, shortcut, activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.block(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNetBasicBlock(ResidualBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        expansion: int = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: Union[int, Tuple[int, int]] = 1,
        activation: nn.Module = nn.ReLU(inplace=True),
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ):
        super().__init__(activation=activation, *args, **kwargs)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.expanded_channels = self.out_channels * expansion

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            elif isinstance(kernel_size, tuple):
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = 0

        self.block = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            activation,
            nn.Conv2d(self.out_channels, self.expanded_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(self.expanded_channels),
        )

        if stride != 1 or dilation != 1 or self.in_channels != self.expanded_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.expanded_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expanded_channels),
            )


class ResNetBottleneckBlock(ResidualBlock):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        bottleneck: int,
        expansion: int = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: Union[int, Tuple[int, int]] = 1,
        activation: nn.Module = nn.ReLU(),
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ):
        super().__init__(activation=activation, *args, **kwargs)
        self.in_channels, self.out_channels = in_channels, in_channels // bottleneck
        self.expanded_channels = self.out_channels * bottleneck * expansion

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            elif isinstance(kernel_size, tuple):
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = 0

        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            activation,
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            activation,
            nn.Conv2d(self.out_channels, self.expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expanded_channels),
        )

        if stride != 1 or dilation != 1 or self.in_channels != self.expanded_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.expanded_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expanded_channels),
            )
