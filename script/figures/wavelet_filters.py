import argparse

import matplotlib.pyplot as plt
import numpy as np
import pywt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", type=str, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=1)
    parser.add_argument("--title", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    wavelet = pywt.Wavelet(args.wavelet)
    levels = np.array([args.levels] if isinstance(args.levels, int) else args.levels, dtype=int)

    _, _, x = wavelet.wavefun(np.max(levels))
    max_len = len(x)
    del x

    fig, axs = plt.subplots(
        len(levels), 1, sharex=True, figsize=(12, 3 * len(levels)), layout="compressed", squeeze=False
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

        t = np.linspace(-max_len // 2, max_len // 2, max_len)
        axs[i, 0].plot(t, psi, marker="o", linewidth=2, label="wavelet filter")
        axs[i, 0].plot(t, phi, marker="o", linewidth=2, label="scaling filter")
        axs[i, 0].set_xlim([-max_len // 2, max_len // 2])
        axs[i, 0].set_ylim([-1.1, 1.1])
        axs[i, 0].set_yticks([-1, -0.5, 0, 0.5, 1])
        axs[i, 0].set_ylabel("normalized magnitude")
        axs[i, 0].legend(loc="upper right")
        if args.title:
            axs[i, 0].set_title(f"level {level}")

    axs[-1, 0].set_xlabel("time [samples]")

    if args.title:
        fig.suptitle(f"impulse response{'s' if len(levels) > 1 else ''}")

    if args.out_path is not None:
        plt.savefig(args.out_path)
    else:
        plt.show()
