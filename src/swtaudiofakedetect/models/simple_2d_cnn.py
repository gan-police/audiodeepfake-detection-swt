import torch
import torch.nn as nn


class Simple2dCNN(nn.Module):  # type: ignore
    """
    Simple 3-layer 2-dimensional CNN.
    Total number of parameters: ~617k
    Expects input shape (1, swt_levels+1, sample_count) = (1, 15, 16384).
    """

    def __init__(self) -> None:
        super().__init__()

        # input: (B, 1, 15, 16384), output: (B, 32, 8, 8192)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, dilation=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        # input: (B, 32, 8, 8192), output: (B, 64, 4, 4096)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        # input: (B, 64, 4, 4096), output: (B, 128, 2, 1024)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=(1, 2), padding=1),
        )

        # input: (B, 128 * 2 * 1024), output: (B, 2)
        self.fc = nn.Linear(128 * 2048, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass"""
        batch_size: int = x.shape[0]
        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out.view(batch_size, -1))
        return out

    def count_parameters(self) -> int:
        """Count the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters())
