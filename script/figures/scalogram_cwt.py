import matplotlib.pyplot as plt
import numpy as np
import pywt
from librosa import load
from matplotlib.patches import Rectangle

from swtaudiofakedetect.plot_utils import plot_cwt

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    path = "downloads/LJSpeech-1.1/wavs/LJ001-0001.wav"
    audio, sr = load(path, sr=22050, dtype=np.float32)

    wavelet = pywt.ContinuousWavelet("shan1.0-1.5", dtype=np.float32)
    x_index_end = 44100

    fig, ax = plt.subplots(1, sharey=True, figsize=(9, 4))

    scales = np.geomspace(0.1, 512, num=256)
    fig, ax, qm, frequencies = plot_cwt(
        audio, int(sr), scales, wavelet, x_index_end=x_index_end, ax=ax, color_bar=False, shading="gouraud"
    )

    inset = ax.inset_axes((0.05, 0.45, 0.4, 0.5))
    data: np.ndarray = np.flip(qm.get_array(), axis=0)

    # define zoom locations
    x_zoom_start, x_zoom_end = 1, 1.25  # in seconds
    y_zoom_start, y_zoom_end = 0.300, 2.100  # in KHz

    rect = Rectangle(
        (x_zoom_start, y_zoom_start),
        x_zoom_end - x_zoom_start,
        y_zoom_end - y_zoom_start,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

    # find position in coefficient array
    x_index_start = int(x_zoom_start * 22050)
    x_index_end = int(x_zoom_end * 22050)
    y_frequencies = np.flip(frequencies)  # put frequencies in correct order from low to high
    y_index_start = np.abs(y_frequencies - y_zoom_start).argmin()
    y_index_end = np.abs(y_frequencies - y_zoom_end).argmin()

    x, y = np.meshgrid(np.arange(x_index_end - x_index_start), np.arange(y_index_end - y_index_start))
    inset.pcolormesh(
        x,
        y,
        data[y_index_start:y_index_end, x_index_start:x_index_end],
        vmin=data.min(),
        vmax=data.max(),
        shading="gouraud",
    )

    for loc in ["top", "bottom", "left", "right"]:
        inset.spines[loc].set_linewidth(2)
        inset.spines[loc].set_color("red")
    inset.tick_params(
        axis="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    fig.tight_layout()
    fig.colorbar(qm, ax=ax, fraction=0.06, aspect=30, pad=0.02, label="decibel [dB]")

    # plt.savefig("out/figures/sample_cwt.png", dpi=300)
    plt.show()
