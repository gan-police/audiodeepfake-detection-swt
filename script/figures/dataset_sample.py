from os.path import basename

import matplotlib.pyplot as plt
import pywt
from librosa import load

from swtaudiofakedetect.plot_utils import plot_scalogram, plot_spectrogram

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    files = [
        "downloads/LJSpeech-1.1/wavs/LJ001-0001.wav",
        "downloads/WaveFake/ljspeech_full_band_melgan/LJ001-0001_gen.wav",
    ]

    for file in files:
        au, sr = load(file)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="compressed")
        fig.set_dpi(300)

        plot_spectrogram(au, int(sr), index_start=16000, index_end=38051, ax=axs[0], title="STFT Spectrogram", vmin=-80)

        plot_scalogram(
            au,
            int(sr),
            "swt",
            wavelet=pywt.Wavelet("db4"),
            x_index_start=10000,
            x_index_end=42768,
            x_crop_left=6000,
            x_crop_right=28051,
            z_unit="magnitude",
            normalize=True,
            ax=axs[1],
            title="SWT Scalogram",
        )

        # plt.savefig(f"out/figures/dataset_sample_{basename(file).replace('.wav', '')}.pdf")
        plt.show()
