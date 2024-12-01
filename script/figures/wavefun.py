import argparse

import matplotlib.pyplot as plt
import numpy as np
import pywt

if __name__ == "__main__":
    """Visualize filter and frequency properties for discrete wavelets.
    Adaption of the continuous wavelet visualization found here:
    https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#converting-frequency-to-scale-for-cwt"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", type=str, required=True)
    parser.add_argument("--levels", type=int, default=3)
    args = parser.parse_args()

    wavelet = pywt.Wavelet(args.wavelet)
    levels = np.arange(1, args.levels + 1)

    _, _, x = wavelet.wavefun(levels[-1])
    max_len = len(x)
    del x

    fig, axs = plt.subplots(len(levels), 2, figsize=(12, 6))
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

        t = np.linspace(-max_len // 2, max_len // 2, max_len)
        axs[i, 0].plot(t, psi, label=r"wavelet filter $\psi$")
        axs[i, 0].plot(t, phi, label=r"scaling filter $\phi$")
        axs[i, 0].set_xlim([-max_len // 2, max_len // 2])
        axs[i, 0].set_ylim([-1, 1])
        axs[i, 0].set_title(f"impulse response for level {level}")

        f = np.linspace(0, np.pi, max_len // 2)
        phi_fft = np.fft.fftshift(np.fft.fft(phi, n=max_len))[(max_len + 1) // 2 :]
        phi_fft /= np.abs(phi_fft).max()
        psi_fft = np.fft.fftshift(np.fft.fft(psi, n=max_len))[(max_len + 1) // 2 :]
        psi_fft /= np.abs(psi_fft).max()
        axs[i, 1].plot(f, np.abs(psi_fft) ** 2, label=r"wavelet filter $\psi$")
        axs[i, 1].plot(f, np.abs(phi_fft) ** 2, label=r"scaling filter $\phi$")
        axs[i, 1].set_xlim([0, np.pi])
        axs[i, 1].set_ylim([0, 1])
        axs[i, 1].set_xticks([0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi])
        axs[i, 1].set_xticklabels(["0", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])
        axs[i, 1].grid(True, axis="x")
        axs[i, 1].set_title(rf"squared gain of frequency response for level {level}")

        axs[i, 0].set_xlabel("time [samples]")
        axs[i, 1].set_xlabel("frequency [radians]")
        axs[0, 0].legend(loc="upper right")
        axs[0, 1].legend(loc="upper right")

    fig.tight_layout()
    plt.show()
