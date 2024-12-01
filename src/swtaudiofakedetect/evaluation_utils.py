from datetime import datetime
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pandas import DataFrame, read_csv
from torch.utils.data import Subset

import swtaudiofakedetect.dataset_transform as dtf
from swtaudiofakedetect.dataset import initialize_dataset
from swtaudiofakedetect.dataset_utils import Generators, References
from swtaudiofakedetect.trainer import Trainer
from swtaudiofakedetect.training_utils import get_device, initialize_model, setup
from swtaudiofakedetect.utils import format_float as ff


def combine_training_progress(in_paths: List[str], out_path: Optional[str] = None) -> Dict[str, Any]:
    """Combine the training progress of the same model trained on different seeds using their respective checkpoints."""
    counter = 0
    epoch: int | None = None

    train_loss_list: List[np.ndarray] = []
    valid_acc_list: List[np.ndarray] = []
    valid_loss_list: List[np.ndarray] = []
    test_acc_list: List[float] = []
    test_loss_list: List[float] = []
    test_eer_list: List[float] = []

    for path in in_paths:
        cp = torch.load(path, map_location=torch.device("cpu"))

        # skip intermediate checkpoints
        if cp["test_acc"] is None or cp["test_loss"] is None or cp["test_eer"] is None:
            continue

        if epoch is None:
            epoch = cp["epoch"]

        # sanity assertions
        assert epoch == cp["epoch"]
        assert len(cp["train_loss_history"]) == len(cp["valid_acc_history"]) == len(cp["valid_loss_history"])

        train_loss_list.append(np.array(cp["train_loss_history"]))
        valid_acc_list.append(np.array(cp["valid_acc_history"]))
        valid_loss_list.append(np.array(cp["valid_loss_history"]))
        test_acc_list.append(cp["test_acc"])
        test_loss_list.append(cp["test_loss"])
        test_eer_list.append(cp["test_eer"])

        counter += 1

    result = {"count": counter}
    if counter > 0:
        train_loss_stacked = np.stack(train_loss_list)
        valid_acc_stacked = np.stack(valid_acc_list)
        valid_loss_stacked = np.stack(valid_loss_list)
        test_acc_stacked = np.stack(test_acc_list)
        test_loss_stacked = np.stack(test_loss_list)
        test_eer_stacked = np.stack(test_eer_list)

        result = result | {
            "epochs": np.arange(0, epoch + 1),
            "train_loss_mean": np.mean(train_loss_stacked, axis=0, where=train_loss_stacked != -1),
            "train_loss_std": np.std(train_loss_stacked, axis=0, where=train_loss_stacked != -1),
            "valid_acc_mean": np.mean(valid_acc_stacked, axis=0, where=valid_acc_stacked != -1),
            "valid_acc_std": np.std(valid_acc_stacked, axis=0, where=valid_acc_stacked != -1),
            "valid_loss_mean": np.mean(valid_loss_stacked, axis=0, where=valid_loss_stacked != -1),
            "valid_loss_std": np.std(valid_loss_stacked, axis=0, where=valid_loss_stacked != -1),
            "test_acc_max": np.max(test_acc_stacked),
            "test_acc_mean": np.mean(test_acc_stacked),
            "test_acc_std": np.std(test_acc_stacked),
            "test_loss_min": np.min(test_loss_stacked),
            "test_loss_mean": np.mean(test_loss_stacked),
            "test_loss_std": np.std(test_loss_stacked),
            "test_eer_min": np.min(test_eer_stacked),
            "test_eer_mean": np.mean(test_eer_stacked),
            "test_eer_std": np.std(test_eer_stacked),
        }

    if out_path is not None:
        torch.save(result, out_path)

    return result


def combine_confusion_matrices(in_paths: List[str], out_path: Optional[str] = None) -> Dict[str, Any]:
    """Combine the confusion matrices of test runs of the same model trained on different seeds."""
    result = {}

    matrices_list: List[np.ndarray] = []

    for path in in_paths:
        cm = torch.load(path, map_location=torch.device("cpu"))

        if "classes" not in result:
            result["classes"] = cm["classes"]

        matrices_list.append(cm["matrix"])

    matrices_stacked = np.stack(matrices_list)

    result["count"] = len(matrices_list)
    result["matrices_mean"] = np.mean(matrices_stacked, axis=0)
    result["matrices_std"] = np.std(matrices_stacked, axis=0)

    if out_path is not None:
        torch.save(result, out_path)

    return result


def combine_evaluation_results(in_results: List[Dict[str, Any]], out_path: Optional[str] = None) -> Dict[str, Any]:
    """Combines the evaluation results on different (reference,generator)-pairs of multiple seeds."""
    intermediate: Dict[str, Dict[str, List[float] | np.ndarray]] = {}

    for tr in in_results:
        for key, values in tr.items():
            if key in intermediate:
                intermediate[key]["acc"].append(values["acc"])
                intermediate[key]["loss"].append(values["loss"])
                intermediate[key]["eer"].append(values["eer"])
            else:
                intermediate[key] = {
                    "acc": [values["acc"]],
                    "loss": [values["loss"]],
                    "eer": [values["eer"]],
                }

    for key in intermediate.keys():
        intermediate[key]["acc"] = np.stack(intermediate[key]["acc"])
        intermediate[key]["loss"] = np.stack(intermediate[key]["loss"])
        intermediate[key]["eer"] = np.stack(intermediate[key]["eer"])

    result = {}
    for key, values in intermediate.items():
        result[key] = {  # key is a (reference,generator)-pair
            "acc_max": np.max(values["acc"]),  # maximum accuracy on the different seeds
            "acc_mean": np.mean(values["acc"]),  # mean accuracy on the different seeds
            "acc_std": np.std(values["acc"]),  # standard deviation of accuracy on different seeds
            "loss_min": np.min(values["loss"]),  # ...
            "loss_mean": np.mean(values["loss"]),
            "loss_std": np.std(values["loss"]),
            "eer_min": np.min(values["eer"]),
            "eer_mean": np.mean(values["eer"]),
            "eer_std": np.std(values["eer"]),
        }

    if out_path is not None:
        torch.save(result, out_path)

    return result


def read_evaluation_results(in_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Reads evaluation results of the same model on different seeds given their respective paths and returns two
    arrays of shape (num_seeds, num_pairs) and a sorted list of the (reference, generator)-pairs. Attention: the arrays
    may include np.NaN values."""
    in_results = [torch.load(p, map_location=torch.device("cpu")) for p in in_paths]
    pairs_max_item = max(in_results, key=lambda k: len(k.keys()))
    pairs_list = sorted(pairs_max_item.keys())

    acc_array = np.full((len(in_results), len(pairs_list)), fill_value=np.NaN)
    eer_array = np.full((len(in_results), len(pairs_list)), fill_value=np.NaN)

    for i, er in enumerate(in_results):
        for er_k, er_v in er.items():
            j = pairs_list.index(er_k)
            acc_array[i, j] = er_v["acc"]
            eer_array[i, j] = er_v["eer"]

    return acc_array, eer_array, pairs_list


def print_evaluation_results(results: Dict[str, Dict[str, float]], out_path: Optional[str] = None) -> None:
    from swtaudiofakedetect.dataset_utils import MAP_GENERATOR_NAMES

    # length of the longest generator name
    name_max_length = max(len(s) for s in MAP_GENERATOR_NAMES.values())

    col_widths = [name_max_length + 4, *([16] * 6)]

    lines: List[str] = [
        f"{'Reference':<12} "
        f"{'Generator':<{col_widths[0]}} "
        f"{'Acc. max':<{col_widths[1]}} "
        f"{'Acc. μ ±σ':<{col_widths[2]}} "
        f"{'Loss min':<{col_widths[3]}} "
        f"{'Loss μ ±σ':<{col_widths[4]}} "
        f"{'EER min':<{col_widths[5]}} "
        f"{'EER μ ±σ':<{col_widths[6]}} "
    ]

    for i, (key, values) in enumerate(sorted(results.items(), key=lambda item: item[0])):
        reference, generator = key.split(":")
        lines.append(
            f"{reference:<12} "
            f"{MAP_GENERATOR_NAMES[generator]:<{col_widths[0]}} "
            f"{ff(values['acc_max'], p=True):<{col_widths[1]}} "
            f"{ff(values['acc_mean'], p=True) + ' ±' + ff(values['acc_std'], p=True):<{col_widths[2]}} "
            f"{ff(values['loss_min']):<{col_widths[3]}} "
            f"{ff(values['loss_mean']) + ' ±' + ff(values['loss_std']):<{col_widths[4]}} "
            f"{ff(values['eer_min']):<{col_widths[5]}} "
            f"{ff(values['eer_mean']) + ' ±' + ff(values['eer_std']):<{col_widths[6]}} "
        )

    if out_path is not None:
        with open(out_path, "w") as f:
            f.writelines(line + "\n" for line in lines)
    else:
        for line in lines:
            print(line)


def create_evaluation_df(train_df: DataFrame, dataset_df: DataFrame, max_per_class: int) -> DataFrame:
    # split training dataset by reals and fakes
    train_reals: DataFrame = train_df.loc[train_df["fake"] == False]
    train_fakes: DataFrame = train_df.loc[train_df["fake"] == True]

    # fake sample ids (file names) that the given model was trained on
    trained_fake_ids: List[str] = train_fakes["name"].to_list()
    trained_fake_ids = [x.replace(".wav", "").replace("_gen", "").replace("erated", "") for x in trained_fake_ids]

    # split the evaluation dataset by reals and fakes
    result_reals: DataFrame = dataset_df.loc[dataset_df["fake"] == False]
    result_fakes: DataFrame = dataset_df.loc[dataset_df["fake"] == True]

    # remove all seen real samples
    result_reals = result_reals[~result_reals.index.isin(train_reals.index)]
    # remove all seen fake samples by their sample ids
    result_fakes = result_fakes.loc[
        ~result_fakes["name"].apply(lambda x: any([x.startswith(y) for y in trained_fake_ids]))
    ]

    # now the result dataframes do not contain any seen samples
    # and for every generator exactly the same sample ids are included
    result_df = dataset_df[dataset_df.index.isin(result_reals.index.union(result_fakes.index))]

    # sorting the result dataframe for the next step
    result_df.sort_values(["reference", "generator", "name"], inplace=True)

    # select at most N samples of every class
    result_df = result_df.groupby(["reference", "generator"], as_index=False).head(max_per_class)

    return result_df


def evaluate_model(
    model: str,
    model_dir: str,
    output_dir: str,
    dataset_type: str,
    dataset_dir: str,
    wavelet: str,
    stop_epoch: int,
    batch_size: int,
    num_workers: int,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    eval_checkpoint: Optional[str] = None,
    train_csv: str = "train_dataset.csv",
    train_norm_mean_std: str = "train_mean_std.pt",
    references: References = References().all(),
    generators: Generators = Generators().all(),
    max_samples_per_class: int = 5000,
    **kwargs,
) -> None:
    logger = setup(0, output_dir, "evaluation.log")  # seed doesn't matter
    start_timestamp = datetime.now()
    logger.log(f"started script, timestamp: {start_timestamp}")

    # load dataset
    if "references" in dataset_kwargs:
        del dataset_kwargs["references"]
    if "generators" in dataset_kwargs:
        del dataset_kwargs["generators"]
    dataset = initialize_dataset(dataset_type, dataset_dir, **(dataset_kwargs if dataset_kwargs is not None else {}))

    # load normalization
    mean, std, permute = torch.load(join(model_dir, train_norm_mean_std), map_location=get_device())

    # initialize model
    checkpoint_file: str
    if eval_checkpoint is not None:
        checkpoint_file = eval_checkpoint
    else:
        checkpoint_file = f"checkpoint_{stop_epoch}.pt"

    if model == "ResNet50":
        raise NotImplementedError("evaluation of a ResNet50 model is not supported")
    elif "Wide" in model:
        dataset.transform = dtf.ComposeWithMode([dtf.RandomSlice(16384), dtf.ToTensor()])  # (16384,)
        gpu_transform = dtf.Compose(
            [
                dtf.CalculateSWT(wavelet, 14),  # (15, B, 16384)
                dtf.Permute((1, 0, 2)),  # (B, 15, 16384)
                dtf.AbsLog(),
                dtf.Normalize(mean, std, permute),
                dtf.Reshape((1, 15, 16384), batch_mode=True) if model != "Wide6l1dCNN" else dtf.NoTransform(),
            ]
        )

        model = initialize_model(model, join(model_dir, checkpoint_file), gpu_transform=gpu_transform)
    elif "Wpt" in model or model == "DCNN":
        dataset.transform = dtf.ComposeWithMode([dtf.RandomSlice(32768), dtf.ToTensor()])  # (32768,)

        # calculate (x,y)-shape after packet transform given a wavelet
        tmp = dtf.CalculateWPT(wavelet, 8)(torch.zeros(8, 32768))  # returns (Y, B=8, X) tensor
        y_size: int = tmp.shape[0]  # packets (frequency bins) dimension
        x_size: int = tmp.shape[2]  # time dimension
        del tmp

        gpu_transform = dtf.Compose(
            [
                dtf.CalculateWPT(wavelet, 8),  # (Y=2**8=256, B, X)
                dtf.Permute((1, 0, 2)),  # (B, Y, X)
                dtf.AbsLog(),
                dtf.Normalize(mean, std, permute),
                dtf.Reshape((1, y_size, x_size), batch_mode=True),  # (B, 1, Y, X)
            ]
        )

        model = initialize_model(model, join(model_dir, checkpoint_file), x_size, y_size, gpu_transform=gpu_transform)
    else:
        raise ValueError(f"unrecognized model name '{model}'")

    # initialize trainer
    trainer = Trainer(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        transforms=dataset.transform,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,  # 'True' does not help since we re-initialize dataloader
        save_directory=None,
    )

    # load train dataset csv
    train_df = read_csv(join(model_dir, train_csv), index_col=0, keep_default_na=False)

    # compute complete evaluation dataframe
    eval_df = create_evaluation_df(train_df, dataset.get_df(), max_samples_per_class)

    results: Dict[str, Dict[str, float]] = {}

    for reference in references.list():
        for generator in generators.list():
            # filter by reference and generator
            eval_df_filtered = eval_df.loc[
                (eval_df["reference"] == reference) & (eval_df["generator"].isin([generator, ""]))
            ]

            eval_reals = len(eval_df_filtered.loc[eval_df_filtered["fake"] == False])
            if eval_reals == 0:
                logger.log(f"dataset does not contain unseen samples for reference set '{reference}', skipping...")
                continue

            eval_fakes = len(eval_df_filtered.loc[eval_df_filtered["fake"] == True])
            if eval_fakes == 0:
                logger.log(f"dataset does not contain unseen samples for generator '{generator}', skipping...")
                continue

            logger.log(
                f"evaluation dataset contains {len(eval_df_filtered.index)} samples, "
                f"counts (real, fake): ({eval_reals}, {eval_fakes})"
            )

            # indices need to be converted to iloc indices with get_indexer method
            test_dataset = Subset(dataset, indices=dataset.get_df().index.get_indexer(eval_df_filtered.index))

            trainer.reinitialize_loader("test", dataset=test_dataset)
            acc, loss, eer = trainer.evaluate(mode="test")
            logger.log(
                f"finished test run for {reference} {generator}: "
                f"acc={round(acc * 100, 2)}%, eer={round(eer, 4):.04f}"
            )

            results[f"{reference}:{generator}"] = {"acc": acc, "loss": loss, "eer": eer}

    torch.save(results, join(output_dir, "evaluation_results.pt"))
    logger.log("saved evaluation results")

    logger.log(f"finished evaluation script, elapsed time: {datetime.now() - start_timestamp}")
