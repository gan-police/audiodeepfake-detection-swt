from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch

from swtaudiofakedetect.dataset import WaveFakeItemMeta


class ConfusionMatrix:
    def __init__(self, df: pandas.DataFrame, get_meta: Callable[[int], WaveFakeItemMeta]) -> None:
        counts = df[["reference", "generator"]].value_counts()
        self.classes: List[Tuple[str, str]] = counts.index.tolist()
        self.matrix = np.zeros((2, len(self.classes)), dtype=int)
        self.get_meta = get_meta

    def add_batch(self, predictions: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> None:
        correct: torch.Tensor = predictions.argmax(dim=1) == labels.argmax(dim=1)

        for i in range(len(correct)):
            meta: WaveFakeItemMeta = self.get_meta(indices[i].item())
            i_class_index = self.classes.index((meta.reference, meta.generator))

            if correct[i]:
                self.matrix[0][i_class_index] += 1
            else:
                self.matrix[1][i_class_index] += 1

    def save(self, save_path: str) -> None:
        torch.save({"classes": self.classes, "matrix": self.matrix}, save_path)

    def load(self, load_path: str) -> None:
        saved = torch.load(load_path)
        self.classes = saved["classes"]
        self.matrix = saved["matrix"]

    def plot(self, save_path: Optional[str] = None) -> None:
        fig = plt.figure()
        fig.set_size_inches(16, 5)

        ax = plt.gca()
        cax = ax.matshow(self.matrix, interpolation="none")
        fig.colorbar(cax)

        # tick labels
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xticks(np.arange(len(self.classes)))
        ax.set_xticklabels(self.classes)
        plt.setp(
            [tick.label2 for tick in ax.xaxis.get_major_ticks()],
            rotation=45,
            ha="left",
            va="center",
            rotation_mode="anchor",
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["correct", "incorrect"])

        # center text
        for (x, y), z in np.ndenumerate(self.matrix.T):
            ax.text(x, y, str(z), ha="center", va="center")

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
