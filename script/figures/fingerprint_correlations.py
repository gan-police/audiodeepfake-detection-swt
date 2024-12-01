import os

import matplotlib.pyplot as plt
import numpy as np
import pywt
from librosa import load
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    directory = "evaluations/fingerprints"

    # select all LJSpeech fingerprints
    files = [
        os.path.join(directory, file) for file in os.listdir(directory) if "ljspeech" in file and file.endswith(".wav")
    ]

    reference_index: int

    for i, path in enumerate(files):
        if "ljspeech.wav" in path:
            reference_index = i
            break

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

    for wavelet in wavelets:
        coeffs = []

        for path in files:
            au, sr = load(path, sr=22050)

            cs = np.stack(pywt.swt(au[: 2**14], wavelet, trim_approx=True))
            cs = np.flip(cs, axis=0)
            coeffs.append(cs)

        ref_coeffs = coeffs.pop(reference_index)
        gen_mean_coeffs = np.mean(np.stack(coeffs), axis=0)

        correlations = np.corrcoef(ref_coeffs, gen_mean_coeffs)

        fig, axs = plt.subplots(1, 1, figsize=(5, 5), layout="compressed")

        im = axs.matshow(correlations, cmap="RdBu", vmin=-1, vmax=1)

        # set grid lines
        axs.xaxis.set_major_locator(FixedLocator([0, 15]))
        axs.yaxis.set_major_locator(FixedLocator([0, 15]))
        axs.set_xticks([x - 0.5 for x in axs.get_xticks()][1:], minor="true")
        axs.set_yticks([x - 0.5 for x in axs.get_yticks()][1:], minor="true")
        axs.grid(which="minor", color="black")

        # set tick labels
        axs.xaxis.set_major_locator(FixedLocator([x for x in range(30) if (x % 15) % 4 == 1]))
        axs.yaxis.set_major_locator(FixedLocator([x for x in range(30) if (x % 15) % 4 == 1]))
        axs.xaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(v % 15 + 1))))
        axs.yaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(v % 15 + 1))))
        axs.tick_params(
            axis="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelbottom=True,
            labeltop=True,
            labelleft=True,
            labelright=True,
        )

        # set rectangles
        # ref_rect = Rectangle((0, 14), 14, -14, linewidth=2, linestyle=":", edgecolor="tab:blue", facecolor="none")
        # axs.add_patch(ref_rect)
        # gen_rect = Rectangle((15, 29), 14, -14, linewidth=2, linestyle=":", edgecolor="tab:red", facecolor="none")
        # axs.add_patch(gen_rect)

        fig.colorbar(im, ax=axs, fraction=0.06, aspect=30, pad=0.02)
        # plt.savefig(f"out/figures/fingerprint_correlations_{wavelet}.pdf", dpi=300)
        plt.show()
