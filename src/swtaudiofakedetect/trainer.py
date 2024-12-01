import os
from typing import Iterable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler

from swtaudiofakedetect.confusion_matrix import ConfusionMatrix
from swtaudiofakedetect.dataset_transform import ComposeWithMode, Transformer
from swtaudiofakedetect.logger import Logger
from swtaudiofakedetect.training_utils import calculate_eer, get_device


class Trainer:
    def __init__(
        self,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        transforms: Transformer,
        train_dataset: Optional[Dataset],
        valid_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        batch_size: int,
        num_workers: int,
        save_directory: str,
        torch_device: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        train_sampler: Union[Sampler, Iterable, None] = None,
        valid_sampler: Union[Sampler, Iterable, None] = None,
        test_sampler: Union[Sampler, Iterable, None] = None,
        train_batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
        valid_batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
        test_batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
        pin_memory: bool = False,
        multiprocessing_context: Optional[str] = None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        scheduler: Optional[LRScheduler] = None,
        logger: Logger = Logger(),
    ):
        self.device = get_device(torch_device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.transforms = transforms

        self.loader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "multiprocessing_context": multiprocessing_context,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }

        if train_dataset is not None:
            self.train_loader = DataLoader(
                dataset=train_dataset,
                shuffle=True,
                sampler=train_sampler,
                batch_sampler=train_batch_sampler,
                **self.loader_args,
            )

        if valid_dataset is not None:
            self.valid_loader = DataLoader(
                dataset=valid_dataset,
                shuffle=False,
                sampler=valid_sampler,
                batch_sampler=valid_batch_sampler,
                **self.loader_args,
            )

        if test_dataset is not None:
            self.test_loader = DataLoader(
                dataset=test_dataset,
                shuffle=False,
                sampler=test_sampler,
                batch_sampler=test_batch_sampler,
                **self.loader_args,
            )

        self.start_epoch: int = 0
        self.current_epoch: int = 0
        self.train_loss_history: List[float] = []
        self.valid_acc_history: List[float] = []
        self.valid_loss_history: List[float] = []
        self.test_acc: Optional[float] = None
        self.test_loss: Optional[float] = None
        self.test_eer: Optional[float] = None

        self.save_directory = save_directory
        self.logger = logger

        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def reinitialize_loader(self, which: Literal["train", "valid", "test"], **kwargs) -> None:
        match which:
            case "train":
                self.train_loader = DataLoader(
                    **(self.loader_args | {"shuffle": True} | kwargs if kwargs is not None else {})
                )
            case "valid":
                self.valid_loader = DataLoader(
                    **(self.loader_args | {"shuffle": False} | kwargs if kwargs is not None else {})
                )
            case "test":
                self.test_loader = DataLoader(
                    **(self.loader_args | {"shuffle": False} | kwargs if kwargs is not None else {})
                )

    def save_checkpoint(self, current_epoch: int) -> None:
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "train_loss_history": self.train_loss_history,
                "valid_acc_history": self.valid_acc_history,
                "valid_loss_history": self.valid_loss_history,
                "test_acc": self.test_acc,
                "test_loss": self.test_loss,
                "test_eer": self.test_eer,
            },
            os.path.join(self.save_directory, f"checkpoint_{current_epoch}.pt"),
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=torch.device("cpu") if not torch.cuda.is_available() else None)
        self.start_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_loss_history = checkpoint["train_loss_history"]
        self.valid_acc_history = checkpoint["valid_acc_history"]
        self.valid_loss_history = checkpoint["valid_loss_history"]
        self.test_acc = checkpoint["test_acc"]
        self.test_loss = checkpoint["test_loss"]
        self.test_eer = checkpoint["test_eer"]

    @torch.no_grad()
    def evaluate(
        self, mode: Literal["validate", "test"] = "validate", confusion_matrix: Optional[ConfusionMatrix] = None
    ) -> Tuple[float, float, Optional[float]]:
        # set dataset transforms to eval mode
        if isinstance(self.transforms, ComposeWithMode):
            self.transforms.set_eval_mode()

        # select the requested data loader
        data_loader: DataLoader
        if mode == "validate":
            data_loader = self.valid_loader
        elif mode == "test":
            data_loader = self.test_loader
        else:
            raise ValueError("mode must either be 'validate' or 'test'")

        correct = 0
        total = 0
        loss_list: List[float] = []
        y_true_list: List[torch.Tensor] = []
        y_score_list: List[torch.Tensor] = []
        test_eer: Optional[float] = None

        # set the model to evaluation mode
        self.model.eval()

        for batch, labels, indices in data_loader:
            batch = batch.to(self.device)
            labels = labels.to(self.device)

            output = self.model.forward(batch)

            if mode == "test":
                y_true_list.append(labels.argmax(dim=1))  # 0 is real, 1 is fake
                y_score_list.append(output.softmax(dim=1)[:, 1])  # probabilities for the positive (-> fake) class

            if confusion_matrix is not None:
                confusion_matrix.add_batch(output, labels, indices)

            loss = self.criterion(output, labels.argmax(dim=1))
            loss_list.append(loss.item())

            correct += torch.where(output.argmax(dim=1) == labels.argmax(dim=1))[0].shape[0]
            total += labels.shape[0]

        if mode == "test":
            test_eer = calculate_eer(torch.cat(y_true_list).cpu(), torch.cat(y_score_list).cpu())

        # set the model back to training mode
        self.model.train()

        # set dataset transforms back to training mode
        if isinstance(self.transforms, ComposeWithMode):
            self.transforms.set_train_mode()

        return correct / total, float(np.mean(loss_list)), test_eer

    def train(
        self,
        stop_epoch: int,
        num_validations: int,
        num_checkpoints: int,
        confusion_matrix: Optional[ConfusionMatrix] = None,
    ) -> None:
        """
        Train the model.

        :param int stop_epoch: Stop training after this epoch
        :param int num_validations: Number of evaluations of the model (excluding the final validation after training)
        :param int num_checkpoints: Number of checkpoints (excluding the final checkpoint after training)
        :param Optional[ConfusionMatrix] confusion_matrix: Store classification results of test run in confusion matrix
        """

        validation_epochs = np.linspace(self.start_epoch + 1, stop_epoch, num_validations + 2, dtype=int)[1:]
        checkpoint_epochs = np.linspace(self.start_epoch + 1, stop_epoch, num_checkpoints + 2, dtype=int)[1:]

        if self.start_epoch == 0:
            # evaluate untrained model
            acc, loss, _ = self.evaluate()
            self.valid_acc_history.append(acc)
            self.valid_loss_history.append(loss)
            self.train_loss_history.append(-1.0)

        for epoch in range(self.start_epoch + 1, stop_epoch + 1):
            self.current_epoch = epoch
            loss_list: List[float] = []

            for i, (batch, labels, _) in enumerate(self.train_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                # clear gradients w.r.t. parameters
                self.optimizer.zero_grad()

                # calculate forward pass
                output = self.model.forward(batch)

                # calculate loss
                loss = self.criterion(output, labels.argmax(dim=1))
                loss_list.append(loss.item())

                # calculate gradients w.r.t. parameters
                loss.backward()

                # updating parameters
                self.optimizer.step()

            # step forward in LR scheduler after finished training
            if self.scheduler is not None:
                self.scheduler.step()

            self.train_loss_history.append(float(np.mean(loss_list)))

            if any(validation_epochs == epoch):
                acc, loss, _ = self.evaluate()
                self.logger.log(f"epoch {epoch} - " f"accuracy: {round(acc * 100, 2)}%, loss: {loss:.02f}")
                self.valid_acc_history.append(acc)
                self.valid_loss_history.append(loss)
            else:
                self.valid_acc_history.append(-1.0)
                self.valid_loss_history.append(-1.0)

            if any(checkpoint_epochs == epoch):
                if epoch == stop_epoch:
                    self.test_acc, self.test_loss, self.test_eer = self.evaluate(
                        mode="test", confusion_matrix=confusion_matrix
                    )
                    self.logger.log(
                        f"test run - "
                        f"accuracy: {round(self.test_acc * 100, 2)}%, loss: {self.test_loss:.02f}, "
                        f"EER: {self.test_eer:.02f}"
                    )
                self.save_checkpoint(epoch)
                self.logger.log(f"epoch {epoch} - saved checkpoint")

    def plot_training_summary(self, confusion_matrix: Optional[ConfusionMatrix] = None) -> None:
        if confusion_matrix is not None:
            confusion_matrix.plot(os.path.join(self.save_directory, "confusion_matrix.png"))

        assert len(self.train_loss_history) == len(self.valid_loss_history) == len(self.valid_acc_history)

        epochs = np.arange(0, len(self.train_loss_history))

        train_loss_history_arr = np.array(self.train_loss_history)
        valid_loss_history_arr = np.array(self.valid_loss_history)
        valid_acc_history_arr = np.array(self.valid_acc_history)

        train_idx = np.where(train_loss_history_arr != -1)[0]
        valid_idx = np.where(valid_loss_history_arr != -1)[0]

        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(16, 5)

        ax[0].set_title("Loss")
        ax[0].plot(epochs[train_idx], train_loss_history_arr[train_idx], label="Training", linewidth=3)
        ax[0].plot(epochs[valid_idx], valid_loss_history_arr[valid_idx], c="red", label="Validation", linewidth=3)
        if self.test_loss is not None:
            ax[0].plot(epochs[-1], self.test_loss, "go", label="Test", linewidth=3)
        ax[0].legend(loc="best")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        ax[1].set_title("Loss (log)")
        ax[1].plot(epochs[train_idx], train_loss_history_arr[train_idx], label="Training", linewidth=3)
        ax[1].plot(epochs[valid_idx], valid_loss_history_arr[valid_idx], c="red", label="Validation", linewidth=3)
        if self.test_loss is not None:
            ax[1].plot(epochs[-1], self.test_loss, "go", label="Test", linewidth=3)
        ax[1].legend(loc="best")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")

        ax[2].set_title("Accuracy")
        ax[2].plot(epochs[valid_idx], valid_acc_history_arr[valid_idx], c="red", label="Validation", linewidth=3)
        if self.test_acc is not None:
            ax[2].plot(epochs[-1], self.test_acc, "go", label="Test", linewidth=3)
        ax[2].legend(loc="best")
        ax[2].set_ylim(top=1)
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Accuracy")

        fig.tight_layout()
        plt.savefig(os.path.join(self.save_directory, "training_progress.png"))
