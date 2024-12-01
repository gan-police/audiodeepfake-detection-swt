from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from swtaudiofakedetect.dataset_transform import Transformer


class WelfordEstimator:
    """Compute the mean and standard deviation using Welford's iterative algorithm. Data is processed in batches
    which enables the computation of very large datasets which do not fit into main memory."""

    def __init__(self, shape: Tuple[int, ...], axis: Optional[Tuple[int, ...]] = None):
        self.axis = axis
        self.permute: Optional[Tuple[int, ...]] = None

        self.count = torch.zeros(1, dtype=torch.float64)
        if axis is None:
            self.mean = torch.zeros(1, dtype=torch.float64)
            self.std = torch.zeros(1, dtype=torch.float64)
            self.m2 = torch.zeros(1, dtype=torch.float64)
        else:
            result_shape = tuple([shape[i] for i in range(len(shape)) if i not in self.axis])
            self.mean = torch.zeros(result_shape, dtype=torch.float64)
            self.std = torch.zeros(result_shape, dtype=torch.float64)
            self.m2 = torch.zeros(result_shape, dtype=torch.float64)

            if any([v > i for i, v in enumerate(self.axis)]):
                self.permute = tuple([*self.axis] + [i for i in range(len(shape)) if i not in self.axis])

    def count_features(self, input_shape: torch.Size) -> torch.Tensor:
        if self.axis is None:
            return torch.prod(torch.tensor(input_shape))
        else:
            return torch.prod(torch.tensor([input_shape[i] for i in range(len(input_shape)) if i in self.axis]))

    def to_device(self, device: torch.device) -> None:
        self.count = self.count.to(device=device)
        self.mean = self.mean.to(device=device)
        self.std = self.std.to(device=device)
        self.m2 = self.m2.to(device=device)

    def update(self, input_data: torch.Tensor) -> None:
        self.to_device(input_data.device)

        self.count += self.count_features(input_data.shape).to(device=input_data.device)
        if self.permute is not None:
            input_data = torch.permute(input_data, self.permute)

        delta1 = torch.sub(input_data, self.mean)
        self.mean += torch.sum(delta1 / self.count, tuple(range(len(self.axis))) if self.axis else None)
        delta2 = torch.sub(input_data, self.mean)
        self.m2 += torch.sum(delta1 * delta2, tuple(range(len(self.axis))) if self.axis else None)

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean, torch.sqrt(self.m2 / self.count)


def calculate_mean_and_std(
    dataset: Dataset,
    axis: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    multiprocessing_context: Optional[str] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    device: Optional[torch.device] = None,
    gpu_transform: Optional[Transformer] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[int, ...]]]:
    data, _, _ = dataset[0]
    data_shape = tuple(data.shape)
    batch_shape = (batch_size, *data_shape)

    welford = WelfordEstimator(shape=batch_shape, axis=axis)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    for data, _, _ in loader:
        data = data.to(device=device)

        if gpu_transform is not None:
            data = gpu_transform(data)

        welford.update(data)

    mean, std = welford.finalize()

    if welford.permute is not None:
        # permute currently includes the batch dimension, needs translation
        welford.permute = tuple([v - 1 for i, v in enumerate(welford.permute) if i > 0])

    return mean, std, welford.permute
