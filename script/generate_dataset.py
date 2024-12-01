import argparse
import os
import sys
from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from time import perf_counter_ns
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas
import pywt

from swtaudiofakedetect.dataset_utils import files_batch_iterator, files_iterator, load_sample, Sample, SampleCSV
from swtaudiofakedetect.utils import print_bytes, print_duration, seed_rngs


def proc_batch(
    dataset_type: str,
    batch_samples: List[Sample],
    sample_rate: int,
    target_length: int,
    swt_wavelet: Optional[pywt.Wavelet] = None,
    swt_levels: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, List[SampleCSV]]:
    data: np.ndarray
    csv: List[SampleCSV] = []

    if dataset_type == "loaded":
        data = np.empty((len(batch_samples), target_length), dtype=np.float64)
    elif dataset_type == "transformed":
        if swt_wavelet is None:
            raise ValueError("parameter SWT wavelet is required")
        elif swt_levels is None:
            raise ValueError("parameter SWT levels is required")

        data = np.empty((len(batch_samples), swt_levels + 1, target_length), dtype=np.float64)
    else:
        raise ValueError("expected dataset type to be either 'loaded' or 'transformed'")

    for i, sample in enumerate(batch_samples):
        loaded = load_sample(sample.path, target_length=target_length, load_sample_rate=sample_rate, **kwargs)

        if dataset_type == "transformed":
            # transform using SWT

            # standard output of pywt.swt is [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]
            # however, with trim_approx=True we get [cAn, cDn, ..., cD2, cD1] ... very convenient
            coeffs = pywt.swt(loaded, swt_wavelet, level=swt_levels, trim_approx=True)

            # stack the coefficients vectors into one matrix
            # stacked.shape = (SWT_LEVELS+1, TARGET_SAMPLE_COUNT)
            stacked = np.stack(coeffs)

            data[i, :] = stacked
        else:
            data[i, :] = loaded

        csv.append(sample.to_sample_csv())

    return data, csv


def main(
    DOWNLOADS_DIRECTORY: str,
    OUTPUT_DIRECTORY: str,
    DATASET_TYPE: str,
    NUM_WORKERS: int,
    BATCH_SIZE: int,
    SEED: int = 42,
    LOAD_SAMPLE_RATE: int = 22050,
    LOAD_SAMPLE_OFFSET: Optional[float] = None,
    LOAD_SAMPLE_DURATION: Optional[float] = None,
    RANDOM_SAMPLE_SLICE: bool = False,
    TARGET_SAMPLE_DURATION: Optional[float] = None,
    TARGET_SAMPLE_LENGTH: Optional[int] = None,
    SWT_WAVELET: str = "haar",
    SWT_LEVELS: Optional[int] = None,
) -> None:
    seed_rngs(SEED)

    DOWNLOADS_DIRECTORY = os.path.abspath(DOWNLOADS_DIRECTORY)
    OUTPUT_DIRECTORY = os.path.abspath(OUTPUT_DIRECTORY)

    # print required arguments
    print(f"--DOWNLOADS_DIRECTORY={DOWNLOADS_DIRECTORY}")
    print(f"--OUTPUT_DIRECTORY={OUTPUT_DIRECTORY}")
    print(f"--NUM_WORKERS={NUM_WORKERS}")
    print(f"--BATCH_SIZE={BATCH_SIZE}")
    print(f"--SEED={SEED}")

    if __debug__:
        from shutil import rmtree

        rmtree(OUTPUT_DIRECTORY)

    if not os.path.isdir(DOWNLOADS_DIRECTORY):
        print(f"directory does not exist: {DOWNLOADS_DIRECTORY}", file=sys.stderr)
        exit(1)

    if not os.path.isdir(OUTPUT_DIRECTORY):
        try:
            os.mkdir(OUTPUT_DIRECTORY)
        except OSError as err:
            print(f"failed to create output directory: {err}", file=sys.stderr)
            exit(1)
    elif len(os.listdir(OUTPUT_DIRECTORY)) > 0:
        print("output directory should be empty, aborting...", file=sys.stderr)
        exit(1)

    START_TIME = datetime.now()

    DATASET_SIZE = 0

    if DATASET_TYPE == "simple":
        # for "simple" type, we compute on main process without worker processes

        dataset_csv: List[SampleCSV] = []
        for sample in files_iterator(DOWNLOADS_DIRECTORY):
            dataset_csv.append(sample.to_sample_csv())
        DATASET_SIZE = len(dataset_csv)

        dataset_df = pandas.DataFrame(dataset_csv, columns=list(SampleCSV._fields))
        dataset_df.to_csv(os.path.join(OUTPUT_DIRECTORY, "dataset.csv"))
    else:
        # set batch kwargs
        BATCH_KWARGS = {"sample_rate": LOAD_SAMPLE_RATE}

        # determine SWT levels and N (SAMPLE_COUNT)
        # see https://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html#multilevel-1d-swt

        if DATASET_TYPE == "transformed" and SWT_LEVELS:
            BATCH_KWARGS["swt_levels"] = SWT_LEVELS
            BATCH_KWARGS["target_length"] = 2**SWT_LEVELS
        elif TARGET_SAMPLE_DURATION:
            BATCH_KWARGS["target_length"] = 2 ** int(np.log2(LOAD_SAMPLE_RATE * TARGET_SAMPLE_DURATION))

            if DATASET_TYPE == "transformed":
                BATCH_KWARGS["swt_levels"] = pywt.swt_max_level(BATCH_KWARGS["target_length"])
        elif TARGET_SAMPLE_LENGTH:
            BATCH_KWARGS["target_length"] = 2
            while BATCH_KWARGS["target_length"] * 2 <= TARGET_SAMPLE_LENGTH:
                BATCH_KWARGS["target_length"] *= 2

            if DATASET_TYPE == "transformed":
                BATCH_KWARGS["swt_levels"] = pywt.swt_max_level(BATCH_KWARGS["target_length"])
        else:
            raise ValueError(
                "dataset type 'loaded' requires TARGET_SAMPLE_DURATION or TARGET_SAMPLE_LENGTH, and "
                "dataset type 'transformed' requires SWT_LEVELS"
            )

        if DATASET_TYPE == "transformed":
            BATCH_KWARGS["swt_wavelet"] = pywt.Wavelet(SWT_WAVELET)

        if LOAD_SAMPLE_OFFSET is not None:
            print(f"--LOAD_SAMPLE_OFFSET={LOAD_SAMPLE_OFFSET}")
            BATCH_KWARGS["load_sample_offset"] = LOAD_SAMPLE_OFFSET
        if LOAD_SAMPLE_DURATION is not None:
            print(f"--LOAD_SAMPLE_DURATION={LOAD_SAMPLE_DURATION}")
            BATCH_KWARGS["load_sample_duration"] = LOAD_SAMPLE_DURATION
        if RANDOM_SAMPLE_SLICE is not None:
            print(f"--RANDOM_SAMPLE_SLICE={RANDOM_SAMPLE_SLICE}")
            BATCH_KWARGS["random_sample_slice"] = RANDOM_SAMPLE_SLICE

        print("BATCH KWARGS:")
        for k, v in BATCH_KWARGS.items():
            if k == "swt_wavelet":
                print(f"{k}={v.name}")
            else:
                print(f"{k}={v}")
        print("")

        start_ns = perf_counter_ns()

        # initialize resulting dataset array and list
        dataset_size = sum([len(batch) for batch in files_batch_iterator(DOWNLOADS_DIRECTORY, BATCH_SIZE)])
        dataset_data: np.ndarray
        dataset_csv: List[SampleCSV] = []
        if DATASET_TYPE == "loaded":
            dataset_data = np.empty((dataset_size, BATCH_KWARGS["target_length"]), dtype=np.float64)
        elif DATASET_TYPE == "transformed":
            dataset_data = np.empty(
                (dataset_size, BATCH_KWARGS["swt_levels"] + 1, BATCH_KWARGS["target_length"]), dtype=np.float64
            )

        # initialize NUM_WORKERS workers
        with ProcessPoolExecutor(NUM_WORKERS) as procs:
            proc_futures: List[Future] = []

            # iterate files in batches
            for batch in files_batch_iterator(DOWNLOADS_DIRECTORY, BATCH_SIZE):
                proc_futures.append(
                    procs.submit(proc_batch, dataset_type=DATASET_TYPE, batch_samples=deepcopy(batch), **BATCH_KWARGS)
                )

            index = 0
            for completed_future in as_completed(proc_futures):
                batch_data, batch_csv = completed_future.result()
                assert len(batch_data) == len(batch_csv)

                dataset_data[index : index + len(batch_data), :] = batch_data
                dataset_csv.extend(batch_csv)
                index += len(batch_data)
            DATASET_SIZE = index

        end_ns = perf_counter_ns()
        print(f"{DATASET_TYPE} all samples, runtime: {print_duration(end_ns - start_ns)}")
        start_ns = perf_counter_ns()

        # create output hdf5 file

        df = pandas.DataFrame(dataset_csv, columns=list(SampleCSV._fields))
        bio = BytesIO()
        df.to_csv(bio)

        # determine hdf5 user block size
        user_block_size = int(2 ** np.ceil(np.log2(bio.getbuffer().nbytes)))

        # create hdf5 file and insert dataset
        with h5py.File(
            os.path.join(OUTPUT_DIRECTORY, "dataset.hdf5"), "w", libver="latest", userblock_size=user_block_size
        ) as h5:
            h5.create_dataset("data", data=dataset_data)

        # insert dataset csv into hdf5 user block
        with open(os.path.join(OUTPUT_DIRECTORY, "dataset.hdf5"), "br+") as h5b:
            h5b.write(bio.getbuffer().tobytes())

        end_ns = perf_counter_ns()
        print(
            f"created hdf5 file, "
            f"size: {print_bytes(os.path.getsize(os.path.join(OUTPUT_DIRECTORY, 'dataset.hdf5')))}, "
            f"user block: {print_bytes(user_block_size)}, "
            f"runtime: {print_duration(end_ns - start_ns)}"
        )

    END_TIME = datetime.now()

    # save dataset metadata
    with open(os.path.join(OUTPUT_DIRECTORY, "metadata.txt"), "w") as file:
        file.write(f"created: {END_TIME.strftime('%Y-%m-%d %H:%M:%S')} (in {str(END_TIME - START_TIME)})\n")
        file.write(f"dataset type: {DATASET_TYPE}\n")
        file.write(f"dataset size: {DATASET_SIZE}\n")
        if DATASET_TYPE == "loaded" or DATASET_TYPE == "transformed":
            file.write(f"hdf5 userblock size: {user_block_size} bytes\n")
            file.write(
                f"sample rate: {LOAD_SAMPLE_RATE}, "
                f"sample length: {BATCH_KWARGS['target_length']} "
                f"(~{BATCH_KWARGS['target_length'] / LOAD_SAMPLE_RATE:.2f}s)\n"
            )
            if RANDOM_SAMPLE_SLICE is True:
                file.write(f"seed: {SEED}\n")
        if DATASET_TYPE == "transformed":
            file.write(
                f"swt wavelet: {BATCH_KWARGS['swt_wavelet'].name}, " f"swt levels: {BATCH_KWARGS['swt_levels']}\n"
            )

    print(f"script completed after {str(END_TIME - START_TIME)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--DOWNLOADS_DIRECTORY", type=str, required=True)
    parser.add_argument("--OUTPUT_DIRECTORY", type=str, required=True)
    parser.add_argument("--DATASET_TYPE", type=str, choices=["simple", "loaded", "transformed"], required=True)
    parser.add_argument("--NUM_WORKERS", type=int, required=True)
    parser.add_argument("--BATCH_SIZE", type=int, required=True)
    parser.add_argument("--SEED", type=int, default=42)
    # wav loading and slicing args (for type={loaded, transformed})
    parser.add_argument("--LOAD_SAMPLE_RATE", type=int, default=22050)
    parser.add_argument("--LOAD_SAMPLE_OFFSET", type=float)
    parser.add_argument("--LOAD_SAMPLE_DURATION", type=float)
    parser.add_argument("--RANDOM_SAMPLE_SLICE", type=bool)
    # targeted sample length in seconds or samples (for type={loaded, transformed})
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--TARGET_SAMPLE_DURATION", type=float, help="desired sample duration in seconds")
    group.add_argument("--TARGET_SAMPLE_LENGTH", type=int, help="desired sample length in samples")
    # wavelet args (for type=transformed)
    parser.add_argument("--SWT_WAVELET", type=str, default="haar")
    group.add_argument("--SWT_LEVELS", type=int)
    args = parser.parse_args()
    main(**vars(args))
