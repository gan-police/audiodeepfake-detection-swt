import os
from typing import Dict, List, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pywt

from swtaudiofakedetect.dataset_utils import load_sample
from swtaudiofakedetect.plot_utils import plot_scalogram, plot_spectrogram, wt_frequencies


class Item(NamedTuple):
    reference: str
    generator: Optional[str]
    wav_data: np.ndarray
    swt_coeffs: List[np.ndarray]


if __name__ == "__main__":
    fingerprints_dir = os.path.abspath("evaluations/fingerprints")

    wav_files = [f for f in os.listdir(fingerprints_dir) if f.endswith(".wav")]

    target_length = 2**15
    load_sample_rate = 22050

    # SWT transform
    swt_wavelet = pywt.Wavelet("haar")
    swt_levels = pywt.swt_max_level(target_length)
    swt_freqs = wt_frequencies(swt_wavelet, swt_levels, load_sample_rate, include_approx=True)

    items: Dict[str, Item] = {}

    # load wav files and make transforms
    for wav_file in wav_files:
        reference_name = wav_file[0 : wav_file.find("_")]
        generator_name = None
        if wav_file.count("_") > 0:
            generator_name = wav_file[wav_file.find("_") + 1 : -4]

        wd = load_sample(
            os.path.join(fingerprints_dir, wav_file), target_length=target_length, load_sample_rate=load_sample_rate
        )

        cl = pywt.swt(wd, swt_wavelet, swt_levels, trim_approx=True)  # [cAn, cDn, ..., cD2, cD1]

        items[wav_file] = Item(reference_name, generator_name, wd, cl)

    for item in items.values():
        if item.generator is not None:
            out_prefix: str = f"{item.reference}_{item.generator}_{swt_wavelet.name}_"

            # plot magnitudes of SWT coefficients

            ref_item: Item = items[item.reference + ".wav"]
            ref_means = np.mean(np.abs(np.stack(ref_item.swt_coeffs)), (1,))
            gen_means = np.mean(np.abs(np.stack(item.swt_coeffs)), (1,))

            y_max = max(ref_means.max(initial=0), gen_means.max(initial=0))

            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(12, 5)

            axs[0].set_title(f"comparison of mean absolute coefficient magnitudes")
            axs[0].plot(swt_freqs, ref_means, marker="o", markersize=5, linestyle="dashed", label=item.reference)
            axs[0].plot(
                swt_freqs,
                gen_means,
                marker="o",
                markersize=5,
                linestyle="dashed",
                label=f"{item.generator.replace('_', ' ')} on {item.reference}",
            )
            axs[0].set_xscale("log")
            axs[0].set_xlim(right=load_sample_rate / 2)
            axs[0].set_yscale("log")
            axs[0].set_ylim([1e-4, y_max + 0.1 * y_max])
            axs[0].set_xlabel("frequency [Hz] (log)")
            axs[0].set_ylabel("coefficient magnitudes (log)")
            axs[0].legend(loc="best")

            axs[1].set_title(f"differences of mean absolute coefficient magnitudes")
            axs[1].plot(swt_freqs, [0 for _ in swt_freqs], color="red")
            axs[1].plot(swt_freqs, ref_means - gen_means, color="black", marker="o", markersize=5, linestyle="dashed")
            axs[1].set_xscale("log")
            axs[1].set_xlim(right=load_sample_rate / 2)
            axs[1].set_ylim([-0.05, 0.05])
            axs[1].set_xlabel("frequency [Hz] (log)")
            axs[1].set_ylabel("differences of coefficient magnitudes")

            fig.tight_layout()
            plt.savefig(os.path.join(fingerprints_dir, f"{out_prefix}coefficients.png"))
            plt.clf()

            # plot STFT spectrograms

            fig, axs = plt.subplots(2, 1)
            fig.set_size_inches(9, 6)

            axs[0].set_title(item.reference)
            plot_spectrogram(ref_item.wav_data, load_sample_rate, index_end=target_length, ax=axs[0], vmax=30)
            axs[1].set_title(item.generator.replace("_", " "))
            plot_spectrogram(item.wav_data, load_sample_rate, index_end=target_length, ax=axs[1], vmax=30)

            fig.tight_layout()
            plt.savefig(os.path.join(fingerprints_dir, f"{out_prefix}spectrogram.png"))
            plt.clf()

            # plot STW scalograms (haar)

            fig, axs = plt.subplots(2, 1)
            fig.set_size_inches(9, 6)

            axs[0].set_title(item.reference)
            plot_scalogram(
                None, load_sample_rate, None, coeffs=np.abs(np.stack(ref_item.swt_coeffs)), y_unit="level", ax=axs[0]
            )
            axs[1].set_title(item.generator.replace("_", " "))
            plot_scalogram(
                None, load_sample_rate, None, coeffs=np.abs(np.stack(item.swt_coeffs)), y_unit="level", ax=axs[1]
            )

            fig.tight_layout()
            plt.savefig(os.path.join(fingerprints_dir, f"{out_prefix}scalogram.png"))
            plt.clf()
