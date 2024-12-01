import matplotlib.colors as mcs
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pywt

from swtaudiofakedetect.dataset_utils import load_sample

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    files = [
        "evaluations/fingerprints/ljspeech.wav",
        "evaluations/fingerprints/ljspeech_melgan.wav",
        "evaluations/fingerprints/ljspeech_multi_band_melgan.wav",
        "evaluations/fingerprints/ljspeech_parallel_wavegan.wav",
    ]

    labels = ["LJSpeech", "MelGAN", "Multi-Band MelGAN", "Parallel WaveGAN"]

    wavelets = [
        "haar",
        "db2",  # D(4) from "Wavelet Methods for Time Series Analysis"
        "db3",  # D(6)
        "db4",  # D(8)
        "db8",
        "coif1",  # C(6)
        "coif4",
        "coif8",
        "sym4",  # LA(8)
        "sym8",
    ]

    colors = list(mcs.TABLEAU_COLORS)

    target_length = 2**15
    load_sample_rate = 22050

    for wavelet in wavelets:
        swt_wavelet = pywt.Wavelet(wavelet)
        swt_levels = pywt.swt_max_level(target_length)
        swt_coeffs = []

        # load and transform fingerprints
        for file in files:
            wd = load_sample(file, target_length=target_length, load_sample_rate=load_sample_rate)

            cl = pywt.swt(wd, swt_wavelet, swt_levels, trim_approx=True)  # [cAn, cDn, ..., cD2, cD1]
            swt_coeffs.append(cl)

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9, 3))

        for i in range(1, len(files)):
            swt_coeffs_ref = swt_coeffs[0]
            swt_coeffs_gen = swt_coeffs[i]

            ref_means = np.mean(np.abs(np.stack(swt_coeffs_ref)), (1,))
            gen_means = np.mean(np.abs(np.stack(swt_coeffs_gen)), (1,))

            x_steps = np.arange(len(swt_coeffs_ref), 0, -1)

            x_labels = [f"cA{swt_levels}"] + [f"cD{level}" for level in range(swt_levels, 0, -1)]

            axs[i - 1].bar(x_steps, np.abs(ref_means - gen_means), color="lightgray", label="absolute difference")
            axs[i - 1].plot(
                x_steps, ref_means, marker="o", markersize=5, linestyle="dashed", color=colors[0], label=labels[0]
            )
            axs[i - 1].plot(
                x_steps, gen_means, marker="o", markersize=5, linestyle="dashed", color=colors[i], label=labels[i]
            )
            axs[i - 1].xaxis.set_major_locator(tck.MaxNLocator(integer=True))
            axs[i - 1].set_ylim(bottom=3e-5, top=3)
            axs[i - 1].set_yscale("log")
            axs[i - 1].set_xlabel("level of decomposition")
            axs[i - 1].set_ylabel("mean coefficient magnitudes (log)")
            axs[i - 1].legend(loc="upper center")

        fig.tight_layout()

        # plt.savefig(f"out/figures/fingerprint_coefficients_{wavelet}.pdf")
        plt.show()
