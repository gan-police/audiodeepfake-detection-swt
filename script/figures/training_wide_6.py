from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from matplotlib.ticker import IndexLocator

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

    directories = ["models/Wide6_1d_JSUT/db4", "models/Wide6_2d_JSUT/db4"]

    colors = ["tab:green", "tab:purple"]

    labels = ["Wide6-1d", "Wide6-2d"]

    fig, axs = plt.subplots(1)
    fig.set_size_inches(6, 5)
    fig.set_dpi(100)

    for i in range(len(directories)):
        data = torch.load(join(directories[i], "training_progress_combined.pt"))

        axs.plot(
            data["epochs"][~np.isnan(data["valid_loss_mean"])],
            data["valid_loss_mean"][~np.isnan(data["valid_loss_mean"])],
            color=colors[i],
            linestyle="solid",
            linewidth=2,
            label="validation",
        )
        axs.fill_between(
            data["epochs"][~np.isnan(data["valid_loss_mean"])],
            data["valid_loss_mean"][~np.isnan(data["valid_loss_mean"])]
            - data["valid_loss_std"][~np.isnan(data["valid_loss_std"])],
            data["valid_loss_mean"][~np.isnan(data["valid_loss_mean"])]
            + data["valid_loss_std"][~np.isnan(data["valid_loss_std"])],
            color=colors[i],
            alpha=0.15,
        )
        axs.plot(data["epochs"][-1], data["test_loss_mean"], "o", markersize=6, color=colors[i], label="test")

    axs.set_xlabel("epoch")
    axs.xaxis.set_major_locator(IndexLocator(base=10, offset=0))
    axs.set_ylabel("mean loss (log)")
    axs.set_yscale("log")
    axs.set_ylim(bottom=0.25)
    legend = axs.legend(loc="upper right")
    axs.add_artist(legend)

    # colors legend
    plt.legend(
        handles=[Patch(facecolor=colors[i], label=X) for i, X in enumerate(["Wide6-1d", "Wide6-2d"])],
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        frameon=False,
        loc="lower left",
        mode="expand",
        ncol=2,
    )

    fig.tight_layout()
    # plt.savefig("out/figures/training_wide_6_loss.pdf")
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
            color=colors[i],
            linestyle="solid",
            linewidth=2,
            label="validation",
        )
        axs.fill_between(
            data["epochs"][~np.isnan(data["valid_acc_mean"])],
            data["valid_acc_mean"][~np.isnan(data["valid_acc_mean"])]
            - data["valid_acc_std"][~np.isnan(data["valid_acc_std"])],
            data["valid_acc_mean"][~np.isnan(data["valid_acc_mean"])]
            + data["valid_acc_std"][~np.isnan(data["valid_acc_std"])],
            color=colors[i],
            alpha=0.15,
        )
        axs.plot(data["epochs"][-1], data["test_acc_mean"], "o", markersize=6, color=colors[i], label="test")

    # axs.set_title("Validation Accuracy")
    axs.set_xlabel("epoch")
    axs.xaxis.set_major_locator(IndexLocator(base=10, offset=0))
    axs.set_ylabel("mean accuracy")
    axs.set_ylim(bottom=0.35, top=1)
    legend = axs.legend(loc="lower right")
    axs.add_artist(legend)

    # colors legend
    plt.legend(
        handles=[Patch(facecolor=colors[i], label=X) for i, X in enumerate(["Wide6-1d", "Wide6-2d"])],
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        frameon=False,
        loc="lower left",
        mode="expand",
        ncol=3,
    )

    fig.tight_layout()
    # plt.savefig("out/figures/training_wide_6_accuracy.pdf")
    plt.show()

    plt.clf()
