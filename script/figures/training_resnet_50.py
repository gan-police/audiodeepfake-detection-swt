from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import IndexLocator

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    directories = [
        "models/ResNet50_JSUT/duplicated_db4",
        "models/ResNet50_pretrained_JSUT/duplicated_db4",
        "models/ResNet50_JSUT/chunked_db4",
        "models/ResNet50_pretrained_JSUT/chunked_db4",
        "models/ResNet50_JSUT/3bands_db4",
        "models/ResNet50_pretrained_JSUT/3bands_db4",
    ]

    colors = [
        "tab:pink",
        "tab:cyan",
        "tab:orange",
    ]

    lines = ["-", "--"]
    markers = ["o", "D"]

    fig, axs = plt.subplots(1)
    fig.set_size_inches(6, 5)
    fig.set_dpi(100)

    for i in range(len(directories)):
        data = torch.load(join(directories[i], "training_progress_combined.pt"))

        axs.plot(
            data["epochs"][~np.isnan(data["valid_loss_mean"])],
            data["valid_loss_mean"][~np.isnan(data["valid_loss_mean"])],
            color=colors[i // 2],
            linestyle=lines[i % 2],
            linewidth=2,
        )
        axs.plot(data["epochs"][-1], data["test_loss_mean"], markers[i % 2], markersize=6, color=colors[i // 2])

        # eb1 = axs[0].errorbar(data["epochs"][-1] + 0.5 + (i // 2) * 0.25 + (i / 3),
        #                       data["test_loss_mean"], data["test_loss_std"],
        #                       color=colors[i % 2], marker="o", linewidth=2)
        # eb1[-1][0].set_linestyle(lines[i // 2])

    # axs.set_title("Validation Loss")
    axs.set_xlabel("epoch")
    axs.xaxis.set_major_locator(IndexLocator(base=4, offset=0))
    axs.set_ylabel("mean loss (log)")
    axs.set_yscale("log")

    # shapes legend
    legend = axs.legend(
        handles=[
            Line2D([0], [0], color="tab:gray", linestyle=lines[0], label="random init, validation"),
            Line2D([0], [0], color="tab:gray", linestyle=lines[1], label="pretrained init, validation"),
            Line2D([0], [0], color="tab:gray", ls="", marker=markers[0], label="random init, test"),
            Line2D([0], [0], color="tab:gray", ls="", marker=markers[1], label="pretrained init, test"),
        ],
        ncol=2,
        loc="best",
    )
    axs.add_artist(legend)

    # colors legend
    plt.legend(
        handles=[Patch(facecolor=colors[i], label=f"ResNet-50 ({X})") for i, X in enumerate(["A", "B", "C"])],
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        frameon=False,
        loc="lower left",
        mode="expand",
        ncol=3,
    )

    fig.tight_layout()
    # plt.savefig("out/figures/training_resnet_50_loss.pdf")
    plt.show()

    plt.clf()

    fig, axs = plt.subplots(1)
    fig.set_size_inches(6, 5)
    fig.set_dpi(100)

    for i in range(len(directories)):
        data = torch.load(join(directories[i], "training_progress_combined.pt"))

        axs.plot(
            data["epochs"][~np.isnan(data["valid_acc_mean"])],
            data["valid_acc_mean"][~np.isnan(data["valid_acc_mean"])],
            color=colors[i // 2],
            linestyle=lines[i % 2],
            linewidth=2,
        )
        axs.plot(data["epochs"][-1], data["test_acc_mean"], markers[i % 2], markersize=6, color=colors[i // 2])

    # axs.set_title("Validation Accuracy")
    axs.set_xlabel("epoch")
    axs.xaxis.set_major_locator(IndexLocator(base=4, offset=0))
    axs.set_ylabel("mean accuracy")
    axs.set_ylim(bottom=0.5, top=1)

    # shapes legend
    legend = axs.legend(
        handles=[
            Line2D([0], [0], color="tab:gray", linestyle=lines[0], label="random init, validation"),
            Line2D([0], [0], color="tab:gray", linestyle=lines[1], label="pretrained init, validation"),
            Line2D([0], [0], color="tab:gray", ls="", marker=markers[0], label="random init, test"),
            Line2D([0], [0], color="tab:gray", ls="", marker=markers[1], label="pretrained init, test"),
        ],
        ncol=2,
        loc="best",
    )
    axs.add_artist(legend)

    # colors legend
    plt.legend(
        handles=[Patch(facecolor=colors[i], label=f"ResNet-50 ({X})") for i, X in enumerate(["A", "B", "C"])],
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        frameon=False,
        loc="lower left",
        mode="expand",
        ncol=3,
    )

    fig.tight_layout()
    # plt.savefig("out/figures/training_resnet_50_accuracy.pdf")
    plt.show()

    plt.clf()
