import argparse

import matplotlib.pyplot as plt
import pywt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", type=str, required=True)
    parser.add_argument("--mode", choices=["plot", "file"], default="plot")
    args = parser.parse_args()

    wavelet = pywt.Wavelet(args.wavelet)
    [_, psi, x] = wavelet.wavefun()

    if args.mode == "file":
        print(f"Creating table.dat for {wavelet.name} with {len(x)} entries...")

        lines = []
        for i in range(len(x)):
            lines.append(f"{x[i]}\t{psi[i]}\n")

        with open(f"out/psi_{wavelet.name}_{len(x)}.dat", "w") as file:
            file.writelines(lines)
            print(f"Wrote data to file.")
    else:
        plt.title(wavelet.name)
        plt.plot(x, psi)
        plt.show()
