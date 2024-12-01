import argparse

import matplotlib.pyplot as plt
import numpy as np
import pywt

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", type=str, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=1)
    parser.add_argument("--fft_n", type=int)
    parser.add_argument("--title", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    wavelet = pywt.Wavelet(args.wavelet)
    levels = np.array([args.levels] if isinstance(args.levels, int) else args.levels, dtype=int)

    _, _, x = wavelet.wavefun(np.max(levels))
    max_len = len(x)
    del x

    fft_n = args.fft_n if args.fft_n is not None else max_len

    fig, axs = plt.subplots(
        len(levels), 1, sharex=True, figsize=(8, 2 * len(levels)), layout="compressed", squeeze=False
    )
    for i, level in enumerate(levels):
        phi, psi, _ = wavelet.wavefun(level)
        assert len(phi) == len(psi)
        cur_len = len(phi)

        # normalize to [-1, 1] for nice plotting
        phi /= np.abs(phi).max()
        psi /= np.abs(psi).max()

        # pad with zeros on both sides
        pad_l, pad_r = (max_len - cur_len) // 2, (max_len - cur_len + 1) // 2
        assert pad_l + cur_len + pad_r == max_len
        phi = np.pad(phi, [pad_l, pad_r], mode="constant")
        psi = np.pad(psi, [pad_l, pad_r], mode="constant")

        f = np.linspace(0, np.pi, fft_n // 2)
        phi_fft = np.fft.fftshift(np.fft.fft(phi, n=fft_n))[(fft_n + 1) // 2 :]
        phi_fft /= np.abs(phi_fft).max()
        psi_fft = np.fft.fftshift(np.fft.fft(psi, n=fft_n))[(fft_n + 1) // 2 :]
        psi_fft /= np.abs(psi_fft).max()
        axs[i, 0].plot(f, np.abs(psi_fft) ** 2, linewidth=2, label="wavelet filter")
        axs[i, 0].plot(f, np.abs(phi_fft) ** 2, linewidth=2, label="scaling filter")
        axs[i, 0].set_xlim([0, np.pi])
        axs[i, 0].set_xticks([0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi])
        axs[i, 0].set_xticklabels(["0", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])
        axs[i, 0].set_ylim([0, 1])
        axs[i, 0].set_ylabel("power")
        axs[i, 0].grid(True, axis="x")
        axs[i, 0].legend(loc="upper right")
        if args.title:
            axs[i, 0].set_title(f"level {level}")

    axs[-1, 0].set_xlabel("normalized frequency [radians]")

    if args.title:
        fig.suptitle(f"squared gain of frequency response{'s' if len(levels) > 1 else ''}")

    if args.out_path is not None:
        plt.savefig(args.out_path)
    else:
        plt.show()
