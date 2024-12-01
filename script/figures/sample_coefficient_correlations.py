import matplotlib.pyplot as plt
import numpy as np
import pywt
from librosa import load
from matplotlib.ticker import FixedLocator, FuncFormatter

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    files = [
        "downloads/LJSpeech-1.1/wavs/LJ001-0001.wav",
        "downloads/WaveFake/ljspeech_full_band_melgan/LJ001-0001_gen.wav",
    ]

    labels = ["LJSpeech", "Full-Band MelGAN"]

    wavelets = ["haar", "db2", "coif1", "sym4"]  # D(4)  # C(6)  # LA(8)

    for wavelet in wavelets:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="compressed")

        for i in range(len(files)):
            au, sr = load(files[i], sr=22050)

            coeffs = np.stack(pywt.swt(au[: 2**14], wavelet, trim_approx=True))
            coeffs = np.flip(coeffs, axis=0)

            correlation = np.corrcoef(coeffs)

            im = axs[i].matshow(correlation, cmap="RdBu", vmin=-1, vmax=1)
            axs[i].xaxis.set_major_locator(FixedLocator([x for x in range(15) if x % 4 == 1]))
            axs[i].yaxis.set_major_locator(FixedLocator([x for x in range(15) if x % 4 == 1]))
            axs[i].xaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(v + 1))))
            axs[i].yaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(v + 1))))
            axs[i].set_title(labels[i])

        fig.colorbar(im, ax=axs, fraction=0.06, aspect=30, pad=0.02)
        # plt.savefig(f"out/figures/sample_coefficient_correlations_{wavelet}.pdf", dpi=300)
        plt.show()
        plt.clf()
