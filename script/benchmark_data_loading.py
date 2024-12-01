import argparse
import json
import os
import re
import subprocess
from time import perf_counter_ns
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import swtaudiofakedetect.dataset_transform as dtf
from swtaudiofakedetect.dataset import initialize_dataset
from swtaudiofakedetect.logger import Logger
from swtaudiofakedetect.training_utils import get_device
from swtaudiofakedetect.utils import BenchmarkContext, KWArgsAppend, print_bytes, seed_rngs


def numactl_show() -> str:
    numactl = subprocess.run(["numactl", "--show"], shell=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    numactl.check_returncode()

    return numactl.stdout.decode("utf-8")


def retrieve_memory_information():
    numastat = subprocess.run(
        ["numastat", "-p", str(os.getpid())], shell=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    numastat.check_returncode()
    lines = numastat.stdout.decode("utf-8").splitlines()

    # parse file
    region = 0
    unit_multiplier = 1
    columns = []
    rows = {}
    for line in lines:
        if "memory usage" in line:
            # header line, get unit of values
            if "(in GBs)" in line:
                unit_multiplier = 1e9
            elif "(in MBs)" in line:
                unit_multiplier = 1e6
            elif "(in KBs)" in line:
                unit_multiplier = 1e3
            region = 1  # mark table header
        elif re.search(r"-+", line):
            region += 1
        elif line != "":
            split = re.split(r"\s{2,}", line)
            if region == 1:
                columns = split[1:]
            else:
                row = {}
                for i in range(1, len(split)):
                    row[columns[i - 1]] = float(split[i]) * unit_multiplier
                rows[split[0]] = row

    return rows


def benchmark_loading(loader: DataLoader, epochs: int, iterations: int, snapshot_path: str) -> Tuple[float, float]:
    performance: List[int] = []

    for e in range(epochs):
        if e == epochs // 2:
            # make a memory snapshot in the middle of the benchmark
            mem_bench = retrieve_memory_information()
            with open(snapshot_path, "w") as file:
                json.dump(mem_bench, file, indent=2)

        counter = 0
        epoch_start = perf_counter_ns()
        for data, labels, _ in loader:
            data.to(device)
            labels.to(device)

            counter += 1
            if counter >= iterations:
                break
        epoch_end = perf_counter_ns()
        performance.append(epoch_end - epoch_start)

    return np.mean(np.array(performance, dtype=float)), np.std(np.array(performance, dtype=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_kwargs", nargs="*", default={}, action=KWArgsAppend)
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--transforms_lib", type=str, choices=["pywt", "ptwt"], default="ptwt")
    parser.add_argument("--transforms_device", type=str, choices=["cpu", "gpu"], required=True)
    parser.add_argument("--torch_device", type=str, default=None)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--num_iterations", type=int, default=12800)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent_workers", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--multiprocessing_context", type=str)
    parser.add_argument("--prefetch_factor", type=int)
    args = parser.parse_args()

    seed_rngs(args.seed)

    if not os.path.exists(os.path.abspath(args.output_dir)):
        try:
            os.makedirs(os.path.abspath(args.output_dir))
        except FileExistsError:
            pass

    proc_id = os.getenv("SLURM_PROCID", default="?")
    task_pid = os.getpid()

    logger = Logger(os.path.join(args.output_dir, f"{proc_id}_{task_pid}_output.log"))

    logger.log(f"CPU Affinity={os.sched_getaffinity(0)}")
    logger.log(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', default='')}")

    # export all environment variables
    with open(os.path.join(args.output_dir, f"{proc_id}_{task_pid}_env.txt"), "w") as file:
        lines = []
        for k, v in os.environ.items():
            lines.append(f"{k}={v}\n")
        file.writelines(lines)

    # export numactl
    with open(os.path.join(args.output_dir, f"{proc_id}_{task_pid}_numactl.txt"), "w") as file:
        file.write(numactl_show())

    # memory info before dataset initialization
    mem_before = retrieve_memory_information()

    with BenchmarkContext("initialized dataset in ", logger.log):
        dataset = initialize_dataset(args.dataset_type, args.dataset_dir, **args.dataset_kwargs)

    # memory info after dataset initialization
    mem_after = retrieve_memory_information()

    logger.log(f"dataset size: {len(dataset)}")
    logger.log(f"dataset numpy array size: {print_bytes(dataset.get_array_size())}")
    logger.log(f"dataset pandas dataframe size: {print_bytes(dataset.get_dataframe_size())}")
    logger.log(f"NUMA total: {print_bytes(int(mem_after['Total']['Total']))}")
    logger.log(f"NUMA total delta: {print_bytes(int(mem_after['Total']['Total']) - int(mem_before['Total']['Total']))}")

    # dump numa usage snapshots
    with open(os.path.join(args.output_dir, f"{proc_id}_{task_pid}_numastat_before.json"), "w") as file:
        json.dump(mem_before, file, indent=2)
    with open(os.path.join(args.output_dir, f"{proc_id}_{task_pid}_numastat_after.json"), "w") as file:
        json.dump(mem_after, file, indent=2)

    device = get_device(args.torch_device)

    # print benchmark configuration
    logger.log("\n--- Benchmark Configuration --- ")
    logger.log(f"transforms using: {args.transforms_lib}")
    logger.log(f"transforms on: {args.transforms_device}")
    logger.log(f"available torch device: {device}")
    logger.log(f"wavelet: {args.wavelet}")
    logger.log(f"batch_sizes={args.batch_sizes}")
    logger.log(f"num_iterations={args.num_iterations}")
    logger.log(f"num_workers={args.num_workers}")
    logger.log(f"pin_memory={args.pin_memory}")
    logger.log(f"multiprocessing_context={args.multiprocessing_context}")
    logger.log(f"prefetch_factor={args.prefetch_factor}")
    logger.log(f"persistent_workers={args.persistent_workers}")

    gpu_transforms = None

    if args.transforms_device == "cpu":
        if args.transforms_lib == "pywt":
            dataset.transform = dtf.Compose(
                [
                    dtf.RandomSlice(16384),  # (16384,)
                    dtf.CalculateSWT(args.wavelet, 14),  # (15, 16384)
                    dtf.ToTensor(),
                    dtf.Reshape((1, 15, 16384)),  # (1, 15, 16384)
                ]
            )
        else:
            dataset.transform = dtf.Compose(
                [
                    dtf.RandomSlice(16384),  # (16384,)
                    dtf.ToTensor(),
                    dtf.CalculateSWT(args.wavelet, 14),  # (15, 16384)
                    dtf.Reshape((1, 15, 16384)),  # (1, 15, 16384)
                ]
            )
    else:
        if args.transforms_lib == "pywt":
            logger.log("Warning: GPU transforms on PyWavelets not support, using PTWT!")

        dataset.transform = dtf.Compose([dtf.RandomSlice(16384), dtf.ToTensor()])  # (16384,)

        gpu_transforms = dtf.Compose(
            [
                dtf.CalculateSWT(args.wavelet, 14),  # (15, B, 16384)
                dtf.Permute((1, 0, 2)),  # (B, 15, 16384)
                dtf.Reshape((1, 15, 16384), batch_mode=True),  # (B, 1, 15, 16384)
            ]
        )

    results: Dict[int, Dict] = {}

    for batch_size in args.batch_sizes:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            multiprocessing_context=args.multiprocessing_context,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else None,
        )

        elapsed: List[int] = []

        for run in range(args.num_runs):
            logger.log(f"batch size {batch_size}, step {run+1}/{args.num_runs}")

            start = perf_counter_ns()

            iterations = 0
            for batch, _, _ in loader:
                batch = batch.to(device)

                if gpu_transforms is not None:
                    batch = gpu_transforms(batch)

                iterations += batch.shape[0]
                if iterations >= args.num_iterations:
                    break

            end = perf_counter_ns()
            elapsed.append(end - start)

        elapsed: np.ndarray = np.array(elapsed, dtype=np.float32)
        results[batch_size] = {
            "mean": np.mean(elapsed),
            "std": np.std(elapsed),
            "min": np.min(elapsed),
            "max": np.max(elapsed),
        }

        # save intermediate results
        torch.save(results, os.path.join(args.output_dir, f"{proc_id}_{task_pid}_results.pt"))

    logger.log("Benchmark completed.")
