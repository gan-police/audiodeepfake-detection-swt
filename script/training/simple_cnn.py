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
    **kwargs,
):
    logger = setup(seed, output_dir)
    start_timestamp = datetime.now()
    logger.log(f"started script, timestamp: {start_timestamp}")

    perf_start = perf_counter_ns()
    dataset = initialize_dataset(dataset_type, dataset_dir, **(dataset_kwargs if dataset_kwargs is not None else {}))

    # split dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])

    # set dataset transform to preprocessing transforms
    dataset.transform = dtf.Compose(
        [dtf.ToTensor(get_device()), dtf.CalculateSWT(wavelet, 14), dtf.AbsLog()]  # (15, 65536)
    )

    # calculate mean and std for normalization
    mean, std, permute = calculate_mean_and_std(
        train_dataset, axis=None, **kwargs  # on the train set only  # over all axes
    )
    logger.log(f"calculated mean={mean.item():.04f}, std={std.item():.04f} for training set")

    # set dataset transform to training transforms
    dataset.transform = dtf.ComposeWithMode(
        [
            dtf.RandomSlice(16384),  # (16384,)
            dtf.ToTensor(get_device()),
            dtf.CalculateSWT(wavelet, 14),  # (15, 16384)
            dtf.AbsLog(),
            dtf.Normalize(mean, std, permute),
            dtf.Reshape((1, 15, 16384)) if model == "2d" else dtf.NoTransform(),
        ]
    )

    logger.log(f"initialized dataset, elapsed time: {print_duration(perf_counter_ns() - perf_start)}")

    model: torch.nn.Module
    if model == "1d":
        from swtaudiofakedetect.models.simple_1d_cnn import Simple1dCNN

        model = Simple1dCNN()
    elif model == "2d":
        from swtaudiofakedetect.models.simple_2d_cnn import Simple2dCNN

        model = Simple2dCNN()
    logger.log(f"created model, number of parameters: {model.count_parameters()}")

    # weight the loss function with inverse class counts (class frequency)
    train_reals = dataset.count_reals(train_dataset.indices)
    train_fakes = dataset.count_fakes(train_dataset.indices)
    weight = torch.tensor(
        [1 / train_reals, 1 / train_fakes]  # weight for the real samples  # weight for the fake samples
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    logger.log(
        f"created CrossEntropyLoss, "
        f"train counts (real, fake): ({train_reals}, {train_fakes}), "
        f"weights (real, fake): {weight}"
    )

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
    parser.add_argument("--model", type=str, choices=["1d", "2d"], required=True)
    kwargs = parse_args_to_kwargs(parser)
    main(**kwargs)
