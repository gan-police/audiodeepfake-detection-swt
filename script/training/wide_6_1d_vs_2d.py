import argparse
import os
from datetime import datetime
from time import perf_counter_ns
from typing import Any, Dict, Literal, Optional

import torch
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import random_split

import swtaudiofakedetect.dataset_transform as dtf
from swtaudiofakedetect.confusion_matrix import ConfusionMatrix
from swtaudiofakedetect.dataset import initialize_dataset
from swtaudiofakedetect.dataset_normalization import calculate_mean_and_std
from swtaudiofakedetect.trainer import Trainer
from swtaudiofakedetect.training_utils import get_device, setup
from swtaudiofakedetect.utils import print_duration


def main(
    model: Literal["1d", "2d"],
    seed: int,
    dataset_type: str,
    dataset_dir: str,
    wavelet: str,
    output_dir: str,
    stop_epoch: int,
    num_validations: int,
    num_checkpoints: int,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    lr_scheduler: bool = False,
    lr_step_size: int = 10,
    lr_gamma: float = 0.1,
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
    dataset.transform = dtf.ComposeWithMode([dtf.RandomSlice(16384), dtf.ToTensor()])  # (16384,)

    logger.log(f"initialized dataset, elapsed time: {print_duration(perf_counter_ns() - perf_start)}")

    model: torch.nn.Module
    if model == "1d":
        from swtaudiofakedetect.models.wide_6_1d_conv import Wide6l1dCNN

        model = Wide6l1dCNN(
            gpu_transform=dtf.Compose(
                [
                    dtf.CalculateSWT(wavelet, 14),  # (15, B, 16384)
                    dtf.Permute((1, 0, 2)),  # (B, 15, 16384)
                    dtf.AbsLog(),
                    dtf.Normalize(mean, std, permute),
                ]
            )
        )
    elif model == "2d":
        from swtaudiofakedetect.models.wide_6_2d_conv import Wide6l2dCNN

        model = Wide6l2dCNN(
            gpu_transform=dtf.Compose(
                [
                    dtf.CalculateSWT(wavelet, 14),  # (15, B, 16384)
                    dtf.Permute((1, 0, 2)),  # (B, 15, 16384)
                    dtf.AbsLog(),
                    dtf.Normalize(mean, std, permute),
                    dtf.Reshape((1, 15, 16384), batch_mode=True),  # (B, 1, 15, 16384)
                ]
            )
        )
    logger.log(f"created model, number of parameters: {model.count_parameters()}")

    # weight the loss function with inverse class counts (class frequency)
    train_reals = dataset.count_reals(train_dataset.indices)
    train_fakes = dataset.count_fakes(train_dataset.indices)
    loss_weight = torch.tensor(
        [1 / train_reals, 1 / train_fakes],  # weight for the real samples  # weight for the fake samples
        dtype=torch.float32,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    logger.log(
        f"using weighted loss, "
        f"train counts (real, fake): ({train_reals}, {train_fakes}), "
        f"weights (real, fake): {loss_weight}"
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    scheduler: Optional[LRScheduler] = None
    if lr_scheduler:
        logger.log(f"using learning rate scheduler, step_size={lr_step_size}, gamma={lr_gamma}")
        scheduler = StepLR(optimizer, lr_step_size, lr_gamma)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        transforms=dataset.transform,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        save_directory=output_dir,
        logger=logger,
        **kwargs,
    )

    cm = ConfusionMatrix(dataset.get_df(), get_meta=dataset.get_meta)

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
    parser.add_argument("--model", type=str, choices=["1d", "2d"], required=True)
    parser.add_argument("--lr_scheduler", type=bool, default=False)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    kwargs = parse_args_to_kwargs(parser)
    main(**kwargs)
