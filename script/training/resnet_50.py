import argparse
import os
from datetime import datetime
from time import perf_counter_ns
from typing import Any, Dict, Literal, Optional

import torch
from torch.utils.data import random_split

import swtaudiofakedetect.dataset_transform as dtf
from swtaudiofakedetect.confusion_matrix import ConfusionMatrix
from swtaudiofakedetect.dataset import initialize_dataset
from swtaudiofakedetect.dataset_normalization import calculate_mean_and_std
from swtaudiofakedetect.models.resnet_50 import WaveResNet50
from swtaudiofakedetect.trainer import Trainer
from swtaudiofakedetect.training_utils import get_device, setup
from swtaudiofakedetect.utils import print_duration


class Reshape1(dtf.TransformerBase):
    """The 'duplicated' reshape for ResNet-50. Input is expected to be a batch of samples.
    Do not use outside this script."""

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        sliced = batch[:, :, 0:225]  # (B, 15, 225)
        interleaved = torch.repeat_interleave(sliced, 15, dim=1)  # (B, 225, 225)
        expanded = interleaved.expand(3, -1, -1, -1)  # (3, B, 225, 225)
        return expanded.permute((1, 0, 2, 3))  # (B, 3, 225, 225)


class Reshape2(dtf.TransformerBase):
    """The 'chunked' reshape for ResNet-50. Input is expected to be a batch of samples.
    Do not use outside this script."""

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        channel_tensors = []
        # returns N/224 tensors of shape (B, 15, 225) given 15 coefficient vectors
        tiling_split = torch.split(batch, 225, dim=2)
        for i in range(3):
            # returns tensor of shape (B, 225, 225)
            ct = torch.cat(tiling_split[i * 15 : (i + 1) * 15], dim=1)
            channel_tensors.append(ct)

        # creates new (color) dimension, returns tensor of shape (B, 3, 225, 225)
        return torch.stack(channel_tensors, dim=1)


class Reshape3(dtf.TransformerBase):
    """The '3bands' reshape for ResNet-50. Input is expected to be a batch of samples.
    Do not use outside this script."""

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        channel_tensors = []
        # returns 3 tensors of shape (B, 5, N) given 15 coefficient vectors
        channel_split = torch.split(batch, 5, dim=1)
        for ct in channel_split:
            # returns N/224 tensors of shape (B, 5, 225)
            tiling_split = torch.split(ct, 225, dim=2)
            # returns tensor of shape (B, 225, 225)
            rt = torch.cat(tiling_split[0:45], dim=1)
            channel_tensors.append(rt)

        # creates new (color) dimension, returns tensor of shape (B, 3, 225, 225)
        return torch.stack(channel_tensors, dim=1)


def get_reshape(mode: Literal["duplicated", "chunked", "3bands"]) -> dtf.Transformer:
    match mode:
        case "duplicated":
            return Reshape1()
        case "chunked":
            return Reshape2()
        case "3bands":
            return Reshape3()


def main(
    reshape: Literal["duplicated", "chunked", "3bands"],
    seed: int,
    dataset_type: str,
    dataset_dir: str,
    wavelet: str,
    output_dir: str,
    stop_epoch: int,
    num_validations: int,
    num_checkpoints: int,
    pretrained: bool = False,
    weighted_loss: bool = False,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    logger = setup(seed, output_dir)
    start_timestamp = datetime.now()
    logger.log(f"started script, timestamp: {start_timestamp}")

    perf_start = perf_counter_ns()
    dataset = initialize_dataset(
        dataset_type, dataset_dir, transform=dtf.ToTensor(), **(dataset_kwargs if dataset_kwargs is not None else {})
    )

    # split dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])
    # save train dataset
    train_df = dataset.get_df().iloc[train_dataset.indices]
    train_df.to_csv(os.path.join(output_dir, "train_dataset.csv"), index=False)

    # calculate mean and std for normalization
    mean, std, permute = calculate_mean_and_std(
        train_dataset,  # on the train set only
        axis=None,  # over all axes
        device=get_device(),
        gpu_transform=dtf.Compose(
            [dtf.CalculateSWT(wavelet, 14), dtf.Permute((1, 0, 2)), dtf.AbsLog()]  # (15, B, 65536)  # (B, 15, 65536)
        ),
        **kwargs,
    )
    logger.log(f"calculated mean={mean.item():.04f}, std={std.item():.04f} for training set")
    # save train mean and std
    torch.save((mean, std, permute), os.path.join(output_dir, "train_mean_std.pt"))

    # set dataset transform to training transforms
    dataset.transform = dtf.ComposeWithMode([dtf.RandomSlice(16384), dtf.ToTensor()])

    logger.log(f"initialized dataset, elapsed time: {print_duration(perf_counter_ns() - perf_start)}")

    gpu_transforms = dtf.Compose(
        [
            dtf.CalculateSWT(wavelet, 14),  # (15, B, 16384)
            dtf.Permute((1, 0, 2)),  # (B, 15, 16384)
            dtf.AbsLog(),
            dtf.Normalize(mean, std, permute),
            get_reshape(reshape),
        ]
    )

    model = WaveResNet50(gpu_transform=gpu_transforms, pretrained=pretrained)
    logger.log(f"created model, number of parameters: {model.count_parameters()}")

    weight = None
    if weighted_loss:
        # weight the loss function with inverse class counts (class frequency)
        train_reals = dataset.count_reals(train_dataset.indices)
        train_fakes = dataset.count_fakes(train_dataset.indices)
        weight = torch.tensor(
            [1 / train_reals, 1 / train_fakes]  # weight for the real samples  # weight for the fake samples
        )
        logger.log(f"using weighted CrossEntropyLoss with weights (real, fake): {weight}")

    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        transforms=dataset.transform,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        save_directory=output_dir,
        logger=logger,
        **kwargs,
    )

    cm = ConfusionMatrix(dataset.get_df(), dataset.get_meta)

    perf_start = perf_counter_ns()
    trainer.train(stop_epoch, num_validations, num_checkpoints, confusion_matrix=cm)
    logger.log(f"training completed, elapsed time: {print_duration(perf_counter_ns() - perf_start)}")

    trainer.plot_training_summary(confusion_matrix=cm)
    logger.log(f"plotted training summary")

    cm.save(os.path.join(output_dir, "confusion_matrix.pt"))
    logger.log(f"saved confusion matrix")

    logger.log(f"finished script, elapsed time: {datetime.now() - start_timestamp}")


if __name__ == "__main__":
    from swtaudiofakedetect.configuration import parse_args_to_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument("--reshape", type=str, choices=["duplicated", "chunked", "3bands"], required=True)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--weighted_loss", type=bool, default=False)
    kwargs = parse_args_to_kwargs(parser)

    main(**kwargs)
