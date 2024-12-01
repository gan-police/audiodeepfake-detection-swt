import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import swtaudiofakedetect.dataset_transform as dtf
from swtaudiofakedetect.dataset import initialize_dataset
from swtaudiofakedetect.utils import KWArgsAppend, seed_rngs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_kwargs", nargs="*", default={}, action=KWArgsAppend)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    seed_rngs(args.seed)

    dataset = initialize_dataset(
        args.dataset_type,
        args.dataset_dir,
        transform=dtf.Compose([dtf.RandomSlice(16384), dtf.ToTensor()]),
        **args.dataset_kwargs,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    gpu_transform = dtf.Compose(
        [dtf.CalculateSWT(args.wavelet, 14), dtf.Permute((1, 0, 2))]  # (15, B, 16384)  # (B, 15, 16384)
    )

    log_transform = dtf.AbsLog()

    # use previously computed mean and std
    norm_transform = dtf.Compose(
        [dtf.AbsLog(), dtf.Normalize(mean=torch.tensor([-1.4713]), std=torch.tensor([1.3988]))]
    )

    hist1 = np.empty((args.num_samples, args.num_bins), dtype=np.float32)
    bins1 = args.num_bins
    hist2 = np.empty((args.num_samples, args.num_bins), dtype=np.float32)
    bins2 = args.num_bins
    hist3 = np.empty((args.num_samples, args.num_bins), dtype=np.float32)
    bins3 = args.num_bins

    for i, (data, _, _) in enumerate(dataloader):
        # transform data using SWT and flatten
        data = torch.flatten(gpu_transform(data))

        # compute histogram for unmodified values
        hist1[i], edges = np.histogram(data.numpy(), bins=bins1)
        if isinstance(bins1, int):
            bins1 = edges

        # compute histogram for log-transformed values
        hist2[i], edges = np.histogram(log_transform(data).numpy(), bins=bins2)
        if isinstance(bins2, int):
            bins2 = edges

        # compute histogram for normalized log-transformed values
        hist3[i], edges = np.histogram(norm_transform(data).numpy(), bins=bins3)
        if isinstance(bins3, int):
            bins3 = edges

        if i + 1 == args.num_samples:
            break

    mean1 = np.mean(hist1, axis=0)
    std1 = np.std(hist1, axis=0)
    width1 = np.mean(np.diff(bins1))
    mean2 = np.mean(hist2, axis=0)
    std2 = np.std(hist2, axis=0)
    width2 = np.mean(np.diff(bins2))
    mean3 = np.mean(hist3, axis=0)
    std3 = np.std(hist3, axis=0)
    width3 = np.mean(np.diff(bins3))

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18, 6), layout="compressed")
    axs[0].bar(bins1[:-1], height=mean1, width=width1, align="edge", yerr=std1, ecolor="red")
    axs[1].bar(bins2[:-1], height=mean2, width=width2, align="edge", yerr=std2, ecolor="red")
    axs[2].bar(bins3[:-1], height=mean3, width=width3, align="edge", yerr=std3, ecolor="red")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[1].yaxis.set_tick_params(labelleft=True)
    axs[2].set_yscale("log")
    axs[2].yaxis.set_tick_params(labelleft=True)

    for i, label in enumerate(["unmodified", "log-transformed", "normalized"]):
        axs[i].set_title(f"Histogram of {label} Coefficients")
        axs[i].set_xlabel("value bins")
        axs[i].set_ylabel("count")

    if args.output_path is not None:
        plt.savefig(args.output_path)
    else:
        plt.show()
