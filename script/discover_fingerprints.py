import argparse
import os
from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from time import perf_counter_ns
from typing import Dict, List, Tuple

import numpy as np
import pywt
import torch
from ptwt.stationary_transform import iswt, swt
from soundfile import write

from swtaudiofakedetect.dataset_utils import files_batch_iterator, load_sample, Sample
from swtaudiofakedetect.utils import print_duration

# Paper: "Do GANs leave artificial fingerprints?" (https://arxiv.org/abs/1812.11842)
# For equation (1) we chose a simple low-pass filter as a "suitable denoising filter",
# because we expect the speech contents to reside in the lower frequencies and
# the GAN artifacts to reside in the higher frequencies of the full frequency spectrum.
# We implemented the low-pass filter with the help of the stationary wavelet transform,
# by replacing the first level detail coefficients with zeros during reconstruction of the signal.


def proc_batch(
    batch_samples: List[Sample],
    target_sample_count: int,
    load_sample_rate: int,
    load_sample_duration: float,
    swt_wavelet: pywt.Wavelet,
    swt_levels: int,
) -> Dict[Tuple[str, str], List[np.ndarray]]:
    residuals: Dict[Tuple[str, str], List[np.ndarray]] = {}

    for sample in batch_samples:
        i_wav = load_sample(
            file_path=sample.path,
            target_length=target_sample_count,
            load_sample_rate=load_sample_rate,
            load_sample_duration=load_sample_duration,
        )

        i_swt_coeffs = swt(torch.from_numpy(i_wav), swt_wavelet, swt_levels)

        # reconstruct the signal using [cAn, cDn, ..., cD1]
        i_original = iswt(i_swt_coeffs, swt_wavelet)

        # cD1 contains the first level detail (wavelet) coefficients
        # -> frequency content of interval [0.5, 1] (top highest frequencies)
        # as "denoising filter" we use a low-pass filter by replacing the cD1 coefficients with zeros

        # reconstruct the signal only using [cAn, cDn, ..., cD2, [0, 0, ..., 0]]
        i_filtered = iswt(i_swt_coeffs[:-1] + [torch.zeros_like(i_swt_coeffs[-1])], swt_wavelet)

        # calculate residual (1)
        i_residual = i_original - i_filtered

        i_key = (sample.reference, sample.generator)

        if i_key not in residuals:
            residuals[i_key] = [i_residual.numpy()]
        else:
            residuals[i_key].append(i_residual.numpy())

    return residuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DOWNLOADS_DIRECTORY", type=str, required=True)
    parser.add_argument("--OUTPUT_DIRECTORY", type=str, default="out")
    parser.add_argument("--NUM_WORKERS", type=int, required=True)
    parser.add_argument("--BATCH_SIZE", type=int, required=True)
    parser.add_argument("--TARGET_SAMPLE_RATE", type=int, default=22050)
    parser.add_argument("--TARGET_SAMPLE_DURATION", type=float, default=1.0)
    parser.add_argument("--SWT_WAVELET", type=str, default="haar")
    parser.add_argument("--SWT_LEVELS", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(os.path.abspath(args.OUTPUT_DIRECTORY)):
        os.makedirs(os.path.abspath(args.OUTPUT_DIRECTORY))

    SWT_WAVELET = pywt.Wavelet(args.SWT_WAVELET)

    # number of SWT levels = number of times N is evenly divisible by two
    TARGET_SAMPLE_COUNT = 2 ** int(np.log2(args.TARGET_SAMPLE_RATE * args.TARGET_SAMPLE_DURATION))
    if TARGET_SAMPLE_COUNT / args.TARGET_SAMPLE_RATE < args.TARGET_SAMPLE_DURATION:
        print(f"changed target sample duration to ~{TARGET_SAMPLE_COUNT / args.TARGET_SAMPLE_RATE:.2f}s")

    SWT_WAVELET = pywt.Wavelet(args.SWT_WAVELET)
    SWT_LEVELS = pywt.swt_max_level(TARGET_SAMPLE_COUNT)
    if 0 < args.SWT_LEVELS < SWT_LEVELS:
        SWT_LEVELS = args.SWT_LEVELS

    fingerprints: Dict[Tuple[str, str], np.ndarray] = {}

    with ProcessPoolExecutor(args.NUM_WORKERS) as procs:
        perf_start = perf_counter_ns()
        proc_futures: List[Future] = []

        for batch in files_batch_iterator(args.DOWNLOADS_DIRECTORY, args.BATCH_SIZE):
            proc_futures.append(
                procs.submit(
                    proc_batch,
                    batch_samples=deepcopy(batch),
                    target_sample_count=TARGET_SAMPLE_COUNT,
                    load_sample_rate=args.TARGET_SAMPLE_RATE,
                    load_sample_duration=args.TARGET_SAMPLE_DURATION,
                    swt_wavelet=SWT_WAVELET,
                    swt_levels=SWT_LEVELS,
                )
            )

        residuals: Dict[Tuple[str, str], List[np.ndarray]] = {}

        for completed_future in as_completed(proc_futures):
            batch_result = completed_future.result()

            for key, val in batch_result.items():
                if key not in residuals:
                    residuals[key] = val
                else:
                    residuals[key].extend(val)

        print(f"received all residuals, elapsed time: {print_duration(perf_counter_ns() - perf_start)}")
        for key, val in residuals.items():
            print(f"({key[0]}, {key[1]}) - number of residuals: {len(val)}")

            # calculate average over all batches (3)
            fingerprints[key] = np.average(np.array(val), axis=0)

    for (ref, gen), fingerprint in fingerprints.items():
        filename = f"{ref}{'_' + gen if gen is not None else ''}.wav"
        write(os.path.join(args.OUTPUT_DIRECTORY, filename), fingerprint, args.TARGET_SAMPLE_RATE)
        print(f"({ref}, {gen}) - saved fingerprint to '{os.path.join(args.OUTPUT_DIRECTORY, filename)}'")
