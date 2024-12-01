import matplotlib.pyplot as plt
import pywt
from librosa import load

from swtaudiofakedetect.plot_utils import plot_scalogram

if __name__ == "__main__":
    files = [
        "evaluations/fingerprints/ljspeech.wav",
        "evaluations/fingerprints/ljspeech_melgan.wav",
        "evaluations/fingerprints/ljspeech_multi_band_melgan.wav",
        "evaluations/fingerprints/ljspeech_parallel_wavegan.wav",
    ]

    wavelet = pywt.Wavelet("sym4")
    levels = pywt.dwt_max_level(32768, wavelet)

    for tf in ["dwt", "swt"]:
        fig, axs = plt.subplots(2, 2, sharey=True)
        fig.set_size_inches(12, 8)
        fig.set_dpi(100)

        for i in range(len(files)):
            au, sr = load(files[i], sr=22050)
            _, _, qm, _ = plot_scalogram(
                au,
                int(sr),
                transform=tf,
                wavelet="sym4",
                levels=levels,
                x_index_end=32768,
                x_crop_right=22051,
                y_unit="level",
                z_unit="magnitude",
                normalize=True,
                ax=axs[i // 2, i % 2],
                color_bar=False,
                vmax=0,
            )

        fig.tight_layout()
        fig.colorbar(qm, ax=axs, fraction=0.06, aspect=30, pad=0.02, label="coefficient magnitude")

        # plt.savefig(f"out/figures/fingerprint_scalograms_{tf}.png")
        plt.show()
        plt.clf()
