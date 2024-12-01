from os.path import basename

import matplotlib.pyplot as plt
from librosa import load

from swtaudiofakedetect.plot_utils import plot_scalogram

if __name__ == "__main__":
    files = [
        "evaluations/fingerprints/ljspeech.wav",
        "evaluations/fingerprints/ljspeech_melgan.wav",
        "evaluations/fingerprints/ljspeech_multi_band_melgan.wav",
        "evaluations/fingerprints/ljspeech_parallel_wavegan.wav",
    ]

    wavelets = ["haar", "db2", "coif1", "sym4"]  # D(4)  # C(6)  # LA(8)

    for file in files:
        fig, axs = plt.subplots(4, 1, sharex=True)
        fig.set_size_inches(12, 10)
        fig.set_dpi(100)

        for i in range(len(wavelets)):
            au, sr = load(file, sr=22050)
            _, _, qm, _ = plot_scalogram(
                au,
                int(sr),
                transform="swt",
                wavelet=wavelets[i],
                x_index_end=32768,
                y_unit="level",
                z_unit="magnitude",
                normalize=True,
                ax=axs[i],
                title=f"SWT using {wavelets[i]}",
                color_bar=False,
                vmax=-2,
            )

        fig.tight_layout()
        fig.colorbar(qm, ax=axs, fraction=0.06, aspect=30, pad=0.02, label="coefficient magnitude")

        # plt.savefig(f"out/figures/fingerprint_compare_wavelets_{basename(file).replace('.wav', '')}.png")
        plt.show()
        plt.clf()
