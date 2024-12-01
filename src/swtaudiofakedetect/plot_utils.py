from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pywt
from librosa import fft_frequencies, power_to_db, stft
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    index_start: int = 0,
    index_end: int = -1,
    window_size: int = 2048,
    hop_length: int = 512,
    crop_frequency_bottom: Optional[float] = None,
    crop_frequency_top: Optional[float] = None,
    use_KHz: bool = True,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    color_bar: bool = True,
    cmap: Union[Colormap, str] = "viridis",
    vmin: float = -80,
    vmax: Optional[float] = None,
) -> Tuple[Figure, Axes, QuadMesh]:
    """Compute the Short-time Fourier transform (STFT) and plot a spectrogram"""

    transformed = stft(audio[index_start:index_end], n_fft=window_size, hop_length=hop_length)

    # first convert into power (use np.abs to cast complex numbers)
    transformed = np.abs(transformed) ** 2
    # then convert to decibel
    transformed = power_to_db(transformed)

    sample_start = len(audio) + index_start if index_start < 0 else index_start
    sample_end = len(audio) + index_end if index_end < 0 else index_end

    steps = np.linspace(sample_start / sample_rate, sample_end / sample_rate, transformed.shape[1])
    frequencies = fft_frequencies(sr=sample_rate, n_fft=window_size)

    y_mask = np.full_like(frequencies, fill_value=True, dtype=bool)
    if crop_frequency_bottom is not None:
        y_mask &= frequencies >= crop_frequency_bottom
    if crop_frequency_top is not None:
        y_mask &= frequencies <= crop_frequency_top
    if use_KHz:
        frequencies /= 1000

    frequencies = frequencies[y_mask]
    transformed = transformed[y_mask]

    x, y = np.meshgrid(steps, frequencies)

    # plotting code

    standalone: bool = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1)
        fig.set_dpi(100)
    else:
        fig = ax.figure

    qm: QuadMesh = ax.pcolormesh(x, y, transformed, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")

    if title is not None:
        ax.set_title(title)

    if color_bar:
        fig.colorbar(qm, ax=ax, label="decibel [dB]")

    ax.set_xlabel("time [sec]")
    ax.set_ylabel(f"frequency [{'K' if use_KHz else ''}Hz]")

    if standalone:
        fig.tight_layout()

    return fig, ax, qm


# compute the frequencies corresponding to the level of the decomposition
# in descending order of level (starting at level=levels), and thus ascending in frequencies
def wt_frequencies(wavelet: pywt.Wavelet, levels: int, sample_rate: int, include_approx: bool = False) -> np.ndarray:
    if include_approx:
        # we want to include the frequency associated with the (final) approximation coefficients
        levels += 1

    frequencies = np.empty((levels,), dtype=np.float64)

    for i in range(levels):
        frequencies[i] = pywt.scale2frequency(wavelet, levels - i)

    dt = 1 / sample_rate  # sampling period (in seconds)
    return frequencies / dt


def normalize_coeffs(coeffs: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(coeffs) + 1e-12)


def plot_coefficients(
    coeffs: np.ndarray,
    sample_rate: int,
    y_unit: Literal["frequency", "level", "scale"],
    frequencies: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    x_index_start: int = 0,
    x_crop_left: Optional[int] = None,  # crop relative to coefficient shape
    x_crop_right: Optional[int] = None,  # crop relative to coefficient shape
    z_unit: Literal["decibel", "magnitude"] = "decibel",
    use_KHz: bool = True,
    normalize: bool = False,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    color_bar: bool = True,
    color_map: str = "viridis",
    shading: Literal["nearest", "gouraud", "auto"] = "gouraud",
    rasterized: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Figure, Axes, QuadMesh, np.ndarray]:
    """Plot a pre-computed time-frequency representation of some data"""

    # sanity assertions
    if y_unit == "frequency":
        if frequencies is None:
            raise ValueError("expected frequencies array")
        elif len(frequencies) != coeffs.shape[0]:
            raise ValueError("given frequencies array does not match coefficient array")
    elif y_unit == "scale":
        if scales is None:
            raise ValueError("expected scales array")
        elif len(scales) != coeffs.shape[0]:
            raise ValueError("given scales array does not match coefficient array")

    # crop coefficients along the time axis if requested
    coeffs_start, coeffs_end = (0, coeffs.shape[1])
    if x_crop_left is not None:
        if x_crop_left < 0:
            coeffs_start = coeffs_end + x_crop_left
        else:
            coeffs_start = x_crop_left
    if x_crop_right is not None:
        if x_crop_right < 0:
            coeffs_end += x_crop_right
        else:
            coeffs_end = x_crop_right
    coeffs = coeffs[:, coeffs_start:coeffs_end]

    if z_unit == "decibel":
        # convert to decibel
        coeffs = power_to_db(np.abs(coeffs) ** 2)
    elif normalize:
        # normalization equivalent to the one in preprocessing
        coeffs = np.log10(np.abs(coeffs) + 1e-12)
        vmin = -12

    x_steps = np.linspace(
        (x_index_start + coeffs_start) / sample_rate, (x_index_start + coeffs_end) / sample_rate, coeffs.shape[1]
    )
    y_steps: np.ndarray
    y_formatter = None
    if y_unit == "frequency":
        y_steps = frequencies / 1000 if use_KHz else frequencies
    elif y_unit == "scale":
        y_steps = scales
    else:
        y_steps = np.arange(coeffs.shape[0])
        y_formatter = lambda y_value, _: int(coeffs.shape[0] - y_value)

    x, y = np.meshgrid(x_steps, y_steps)

    # plotting code

    standalone: bool = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1)
        fig.set_dpi(100)
    else:
        fig = ax.figure

    qm: QuadMesh = ax.pcolormesh(
        x, y, coeffs, vmin=vmin, vmax=vmax, cmap=color_map, shading=shading, rasterized=rasterized
    )

    if title is not None:
        ax.set_title(title)

    if color_bar:
        fig.colorbar(qm, ax=ax, label="decibel [dB]" if z_unit == "decibel" else "coefficient magnitude")

    ax.set_xlabel("time [sec]")
    ax.set_ylabel(f"frequency [{'K' if use_KHz else ''}Hz]" if y_unit == "frequency" else y_unit)

    if y_formatter is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

    if standalone:
        fig.tight_layout()

    return fig, ax, qm, y_steps


def plot_cwt(
    audio: np.ndarray,
    sample_rate: int,
    scales: np.ndarray,
    wavelet: Union[str, pywt.ContinuousWavelet],
    x_index_start: int = 0,  # start index of the sliced input signal to be transformed
    x_index_end: Optional[int] = None,  # end index of the sliced input signal to be transformed
    y_unit: Literal["frequency", "scale"] = "frequency",
    *args: Tuple[Any],
    **kwargs: Dict[str, Any],
) -> Tuple[Figure, Axes, QuadMesh, np.ndarray]:
    # slice audio with optionally given indices
    audio = audio[x_index_start:x_index_end]

    # compute CWT
    coeffs, frequencies = pywt.cwt(audio, scales, wavelet, sampling_period=1 / sample_rate, method="fft")

    # mask frequencies above Nyquist frequency
    y_mask = frequencies <= sample_rate / 2

    return plot_coefficients(
        coeffs[y_mask],
        sample_rate,
        y_unit,
        frequencies=frequencies[y_mask],
        scales=scales[y_mask],
        x_index_start=x_index_start,
        *args,
        **kwargs,
    )


def plot_scalogram(
    audio: Optional[np.ndarray],
    sample_rate: int,  # sample rate of the original input signal
    transform: Optional[Literal["dwt", "dwpt", "swt"]],
    wavelet: Union[str, pywt.Wavelet],
    coeffs: Optional[np.ndarray] = None,  # already computed coefficients (2-dimensional array)
    x_index_start: int = 0,  # start index of the sliced input signal to be transformed
    x_index_end: Optional[int] = None,  # end index of the sliced input signal to be transformed
    levels: Optional[int] = None,
    y_unit: Literal["frequency", "level"] = "level",
    *args,
    **kwargs,
) -> Tuple[Figure, Axes, QuadMesh, np.ndarray]:
    """Compute the specified wavelet transform and plot the output coefficients as a scalogram"""

    if coeffs is None:
        # slice audio with optionally given indices
        audio = audio[x_index_start:x_index_end]

        if isinstance(wavelet, str):
            wavelet = pywt.Wavelet(wavelet)

        if levels is None and transform == "dwt":
            levels = pywt.dwt_max_level(len(audio), wavelet)
        elif levels is None and transform == "swt":
            levels = pywt.swt_max_level(len(audio))

        assert levels is not None and levels > 0

        # [cA_n, cD_n, cD_n-1, ..., cD2, cD1]
        coeffs_list: List[np.ndarray]

        if transform == "dwt":
            coeffs_list = pywt.wavedec(audio, wavelet, level=levels)
            # bring all arrays in coefficient list to the same shape
            for i, coeffs in enumerate(coeffs_list):
                level = levels if i == 0 else levels - i + 1
                repeated = np.repeat(coeffs, 2**level)
                overflow = len(repeated) - len(audio)
                repeated = repeated[overflow // 2 : -overflow // 2]
                assert len(repeated) == len(audio)
                coeffs_list[i] = repeated
        elif transform == "dwpt":
            raise NotImplementedError()
        elif transform == "swt":
            coeffs_list = pywt.swt(audio, wavelet, level=levels, trim_approx=True)
        else:
            raise ValueError("transform must be 'dwt', 'dwpt' or 'swt'")

        # vertically stack the coefficients
        coeffs = np.stack(coeffs_list)  # -> coeffs[0] = cA_n; coeffs[1] = cD_n; coeffs[-1] = cD_1; ...

    if y_unit == "frequency":
        raise NotImplementedError()

    return plot_coefficients(coeffs, sample_rate, y_unit, x_index_start=x_index_start, *args, **kwargs)


def plot_training_progress(combined_training_progress: Dict[str, Any], out_path: Optional[str] = None) -> None:
    epochs: np.ndarray = combined_training_progress["epochs"]

    train_loss_mean: np.ndarray = combined_training_progress["train_loss_mean"]
    train_loss_std: np.ndarray = combined_training_progress["train_loss_std"]

    valid_acc_mean: np.ndarray = combined_training_progress["valid_acc_mean"]
    valid_acc_std: np.ndarray = combined_training_progress["valid_acc_std"]
    valid_loss_mean: np.ndarray = combined_training_progress["valid_loss_mean"]
    valid_loss_std: np.ndarray = combined_training_progress["valid_loss_std"]

    test_acc_mean: float = combined_training_progress["test_acc_mean"]
    test_acc_std: float = combined_training_progress["test_acc_std"]
    test_loss_mean: float = combined_training_progress["test_loss_mean"]
    test_loss_std: float = combined_training_progress["test_loss_std"]

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(16, 5)

    ax[0].set_title("Loss")
    ax[0].plot(
        epochs[~np.isnan(train_loss_mean)],
        train_loss_mean[~np.isnan(train_loss_mean)],
        color="tab:blue",
        label="Training",
        linewidth=3,
    )
    ax[0].fill_between(
        epochs[~np.isnan(train_loss_mean)],
        train_loss_mean[~np.isnan(train_loss_mean)] - train_loss_std[~np.isnan(train_loss_std)],
        train_loss_mean[~np.isnan(train_loss_mean)] + train_loss_std[~np.isnan(train_loss_std)],
        color="tab:blue",
        alpha=0.2,
    )
    ax[0].plot(
        epochs[~np.isnan(valid_loss_mean)],
        valid_loss_mean[~np.isnan(valid_loss_mean)],
        color="tab:red",
        label="Validation",
        linewidth=3,
    )
    ax[0].fill_between(
        epochs[~np.isnan(valid_loss_mean)],
        valid_loss_mean[~np.isnan(valid_loss_mean)] - valid_loss_std[~np.isnan(valid_loss_std)],
        valid_loss_mean[~np.isnan(valid_loss_mean)] + valid_loss_std[~np.isnan(valid_loss_std)],
        color="tab:red",
        alpha=0.2,
    )
    ax[0].plot(epochs[-1], test_loss_mean, "go", label="Test")
    ax[0].plot(epochs[-1], test_loss_mean - test_loss_std, "g^")
    ax[0].plot(epochs[-1], test_loss_mean + test_loss_std, "gv")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend(loc="best")

    ax[1].set_title("Loss (log)")
    ax[1].plot(
        epochs[~np.isnan(train_loss_mean)],
        train_loss_mean[~np.isnan(train_loss_mean)],
        color="tab:blue",
        label="Training",
        linewidth=3,
    )
    ax[1].fill_between(
        epochs[~np.isnan(train_loss_mean)],
        train_loss_mean[~np.isnan(train_loss_mean)] - train_loss_std[~np.isnan(train_loss_std)],
        train_loss_mean[~np.isnan(train_loss_mean)] + train_loss_std[~np.isnan(train_loss_std)],
        color="tab:blue",
        alpha=0.2,
    )
    ax[1].plot(
        epochs[~np.isnan(valid_loss_mean)],
        valid_loss_mean[~np.isnan(valid_loss_mean)],
        color="tab:red",
        label="Validation",
        linewidth=3,
    )
    ax[1].fill_between(
        epochs[~np.isnan(valid_loss_mean)],
        valid_loss_mean[~np.isnan(valid_loss_mean)] - valid_loss_std[~np.isnan(valid_loss_std)],
        valid_loss_mean[~np.isnan(valid_loss_mean)] + valid_loss_std[~np.isnan(valid_loss_std)],
        color="tab:red",
        alpha=0.2,
    )
    ax[1].plot(epochs[-1], test_loss_mean, "go", label="Test")
    ax[1].plot(epochs[-1], test_loss_mean - test_loss_std, "g^")
    ax[1].plot(epochs[-1], test_loss_mean + test_loss_std, "gv")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss (log)")
    ax[1].legend(loc="best")

    ax[2].set_title("Accuracy")
    ax[2].plot(
        epochs[~np.isnan(valid_acc_mean)],
        valid_acc_mean[~np.isnan(valid_acc_mean)],
        color="tab:red",
        label="Validation",
        linewidth=3,
    )
    ax[2].fill_between(
        epochs[~np.isnan(valid_acc_mean)],
        valid_acc_mean[~np.isnan(valid_acc_mean)] - valid_acc_std[~np.isnan(valid_acc_std)],
        valid_acc_mean[~np.isnan(valid_acc_mean)] + valid_acc_std[~np.isnan(valid_acc_std)],
        color="tab:red",
        alpha=0.2,
    )
    ax[2].plot(epochs[-1], test_acc_mean, "go", label="Test")
    ax[2].plot(epochs[-1], test_acc_mean - test_acc_std, "g^")
    ax[2].plot(epochs[-1], test_acc_mean + test_acc_std, "gv")
    ax[2].set_ylim(top=1)
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("accuracy")
    ax[2].legend(loc="best")

    fig.tight_layout()

    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()


def plot_test_matrix(combined_test_results: Dict[str, Any], out_path: Optional[str] = None):
    from swtaudiofakedetect.dataset_utils import MAP_GENERATOR_NAMES

    generators = list(combined_test_results.keys())
    generators.sort()

    acc = np.empty((2, len(generators)))
    eer = np.empty((2, len(generators)))

    for i, generator in enumerate(generators):
        acc[0, i] = combined_test_results[generator]["acc_mean"]
        acc[1, i] = combined_test_results[generator]["acc_std"]
        eer[0, i] = combined_test_results[generator]["eer_mean"]
        eer[1, i] = combined_test_results[generator]["eer_std"]

    fig, axs = plt.subplots(2, 1, figsize=(8, 5), layout="compressed")
    axs[0].set_title("Accuracy")
    mat1 = axs[0].matshow(acc, vmin=0, vmax=1)
    for (x, y), z in np.ndenumerate(acc.T):
        axs[0].text(x, y, "{:.4f}".format(z), ha="center", va="center")
    axs[0].set_xticks(np.arange(len(generators)))
    axs[0].set_xticklabels([MAP_GENERATOR_NAMES[g] for g in generators], rotation=15)
    axs[0].set_yticks(np.arange(2))
    axs[0].set_yticklabels(["mean", "std"], rotation=90)

    axs[1].set_title("Equal Error Rate (EER)")
    mat2 = axs[1].matshow(eer, vmin=0, vmax=1)
    for (x, y), z in np.ndenumerate(eer.T):
        axs[1].text(x, y, "{:.4f}".format(z), ha="center", va="center")
    axs[1].set_xticks(np.arange(len(generators)))
    axs[1].set_xticklabels([MAP_GENERATOR_NAMES[g] for g in generators], rotation=15)
    axs[1].set_yticks(np.arange(2))
    axs[1].set_yticklabels(["mean", "std"], rotation=90)

    fig.colorbar(mat1, fraction=0.06, pad=0.01)
    fig.colorbar(mat2, fraction=0.06, pad=0.01)

    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
