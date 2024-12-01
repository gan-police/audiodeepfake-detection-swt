import matplotlib.pyplot as plt
from librosa import load

from swtaudiofakedetect.plot_utils import plot_spectrogram

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    files = [
        "evaluations/fingerprints/ljspeech.wav",
        "evaluations/fingerprints/ljspeech_melgan.wav",
        "evaluations/fingerprints/ljspeech_full_band_melgan.wav",
        "evaluations/fingerprints/ljspeech_hifiGAN.wav",
    ]

    titles = ["LJSpeech (reference set)", "MelGAN", "Full-Band MelGAN", "HiFi-GAN"]

    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(9, 6))

    for i in range(len(files)):
        au, sr = load(files[i], sr=22050)
        _, _, qm = plot_spectrogram(
            au, int(sr), index_end=22051, ax=axs[i // 2, i % 2], title=titles[i], color_bar=False, vmax=0
        )

    fig.tight_layout()
    fig.colorbar(qm, ax=axs, fraction=0.06, aspect=30, pad=0.02, label="decibel [dB]")

    # plt.savefig("out/figures/fingerprint_spectrograms.png", dpi=300)
    plt.show()
