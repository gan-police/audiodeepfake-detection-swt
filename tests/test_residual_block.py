from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from swtaudiofakedetect.residual_block import ResidualBlock, ResNetBasicBlock, ResNetBottleneckBlock


class TestResidualBlock:
    @pytest.mark.parametrize("fill_value", [1, 2, -1])
    def test_residual_block(self, fill_value: int) -> None:
        # default ResidualBlock only has identity modules and ReLU activation
        block = ResidualBlock()

        # expected value = ReLU(Identity + Identity)
        expected_value = max(0, fill_value + fill_value)

        test_data = torch.full((1, 32, 224, 224), fill_value, dtype=torch.float32)
        test_result = block(test_data)

        assert tuple(test_result.shape) == (1, 32, 224, 224)
        assert torch.equal(test_result, torch.full((1, 32, 224, 224), expected_value, dtype=torch.float32))

    @pytest.mark.parametrize("in_out_channels", [(1, 2), (2, 4), (16, 32)])
    @pytest.mark.parametrize("expansion", [1, 2])
    def test_resnet_block(self, in_out_channels: Tuple[int, int], expansion: int) -> None:
        in_channels: int = in_out_channels[0]
        out_channels: int = in_out_channels[1]

        block = ResNetBasicBlock(
            in_channels=in_channels, out_channels=out_channels, expansion=expansion, kernel_size=3, padding=1
        )

        test_data = torch.ones((1, in_channels, 224, 224))
        test_result = block(test_data)

        assert tuple(test_result.shape) == (1, out_channels * expansion, 224, 224)

    @pytest.mark.parametrize("in_out_channels", [(8, 8), (8, 16)])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_compare_basic(self, in_out_channels: Tuple[int, int], stride: int) -> None:
        in_channels, out_channels = in_out_channels

        shortcut = None
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.01)

        torch_basic = BasicBlock(in_channels, out_channels, stride=stride, downsample=shortcut)
        basic = ResNetBasicBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        torch_basic.apply(init_weights)
        basic.apply(init_weights)

        test_data = torch.ones((1, in_channels, 224, 224), dtype=torch.float32)
        torch_result = torch_basic.forward(test_data)
        result = basic.forward(test_data)

        assert torch_result.shape == result.shape
        assert torch.allclose(torch_result, result)

    @pytest.mark.parametrize("bottleneck", [2, 4])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_compare_bottleneck(self, bottleneck: int, stride: int, dilation: int) -> None:
        expansion = 4
        in_channels = 8
        out_channels = in_channels * expansion

        shortcut = None
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.01)

        torch_bottleneck = Bottleneck(
            in_channels, in_channels, base_width=64 // bottleneck, stride=stride, dilation=dilation, downsample=shortcut
        )
        torch_bottleneck.expansion = expansion
        bottleneck = ResNetBottleneckBlock(
            in_channels,
            kernel_size=3,
            bottleneck=bottleneck,
            expansion=expansion,
            stride=stride,
            padding=dilation,  # to match pytorch conv3x3 implementation
            dilation=dilation,
        )
        torch_bottleneck.apply(init_weights)
        bottleneck.apply(init_weights)

        test_data = torch.ones((1, in_channels, 224, 224), dtype=torch.float32)
        torch_result = torch_bottleneck.forward(test_data)
        result = bottleneck.forward(test_data)

        assert torch_result.shape == result.shape
        assert torch.allclose(torch_result, result)
