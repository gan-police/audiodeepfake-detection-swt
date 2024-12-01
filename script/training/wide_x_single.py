import argparse
import os
from datetime import datetime
from time import perf_counter_ns
from typing import Any, Dict, List, Literal, Optional

import torch
from pandas import DataFrame
from torch.utils.data import random_split

import swtaudiofakedetect.dataset_transform as dtf
from swtaudiofakedetect.confusion_matrix import ConfusionMatrix
from swtaudiofakedetect.dataset import initialize_dataset
from swtaudiofakedetect.dataset_normalization import calculate_mean_and_std
from swtaudiofakedetect.dataset_utils import Generators
from swtaudiofakedetect.trainer import Trainer
from swtaudiofakedetect.training_utils import get_device, initialize_model, setup
from swtaudiofakedetect.utils import print_duration


def main(
    model: Literal["Wide6", "Wide16Basic", "Wide24Basic", "Wide19Bottle", "Wide32Bottle"],
    seed: int,
    dataset_type: str,
    dataset_dir: str,
    wavelet: str,
    output_dir: str,
    stop_epoch: int,
    num_validations: int,
    num_checkpoints: int,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    weighted_loss: bool = False,
    lr_scheduler: bool = True,
    lr_milestones: Optional[List[int]] = None,
    lr_gamma: float = 0.1,
    **kwargs,
):
    logger = setup(seed, output_dir)
    start_timestamp = datetime.now()
    logger.log(f"started script, timestamp: {start_timestamp}")

    train_generator = Generators(dataset_kwargs["generators"])
    if len(train_generator) != 1:
        raise ValueError("this training script expects to train on a single generator")

    perf_start = perf_counter_ns()
    dataset = initialize_dataset(
        dataset_type, dataset_dir, transform=dtf.ToTensor(), **(dataset_kwargs if dataset_kwargs is not None else {})
    )

    # split dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])
    # save train dataset
    train_df: DataFrame = dataset.get_df().iloc[train_dataset.indices]
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

    gpu_transforms = dtf.Compose(
        [
            dtf.CalculateSWT(wavelet, 14),  # (15, B, 16384)
            dtf.Permute((1, 0, 2)),  # (B, 15, 16384)
            dtf.AbsLog(),
            dtf.Normalize(mean, std, permute),
            dtf.Reshape((1, 15, 16384), batch_mode=True),  # (B, 1, 15, 16384)
        ]
    )

    model = initialize_model(model, None, gpu_transforms)
    logger.log(f"created model, number of parameters: {model.count_parameters()}")

    loss_weight = None
    if weighted_loss:
        # weight the loss function with inverse class counts (class frequency)
        train_reals = dataset.count_reals(train_dataset.indices)
        train_fakes = dataset.count_fakes(train_dataset.indices)
        loss_weight = torch.tensor(
            [1 / train_reals, 1 / train_fakes]  # weight for the real samples  # weight for the fake samples
        )
        logger.log(
            f"using weighted loss, "
            f"train counts (real, fake): ({train_reals}, {train_fakes}), "
            f"weights (real, fake): {loss_weight}"
        )

    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if lr_scheduler and lr_milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["Wide6", "Wide16Basic", "Wide24Basic", "Wide19Bottle", "Wide32Bottle"],
    )
    parser.add_argument("--weighted_loss", type=bool, default=False)
    parser.add_argument("--lr_scheduler", type=bool, default=True)
    parser.add_argument("--lr_milestones", type=int, nargs="+")
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    kwargs = parse_args_to_kwargs(parser)

    main(**kwargs)
