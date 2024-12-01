import argparse
import os
import os.path as path

import numpy as np

from swtaudiofakedetect.dataset_utils import get_wavefake_extended_generators, get_wavefake_original_generators
from swtaudiofakedetect.evaluation_utils import read_evaluation_results
from swtaudiofakedetect.utils import format_float as ff

if __name__ == "__main__":
    """Read the evaluation results from a model and pretty print them. Expects 'directory' to contain subdirectories,
    each containing a trained model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--evaluation_results_filename", type=str, default="evaluation_results.pt")
    parser.add_argument("--which_dataset", choices=["original", "extended"], default="original")
    parser.add_argument("--include_JSUT", type=bool, default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    print(
        "\nConfiguration:\n"
        f"\tDataset: {args.which_dataset}\n"
        f"\tInclude JSUT?: {'yes' if args.include_JSUT else 'no'}\n"
        "\n----------------------------------------------------------------"
    )

    todos = sorted(
        [path.join(args.directory, d) for d in os.listdir(args.directory) if path.isdir(path.join(args.directory, d))]
    )

    for todo in todos:
        print(f"\nDir: '{todo}'")

        er_paths = [
            path.join(todo, d, "evaluation_results.pt") for d in os.listdir(todo) if path.isdir(path.join(todo, d))
        ]
        er_ACCs, er_EERs, er_pairs = read_evaluation_results(er_paths)

        acc_where = np.full(er_ACCs.shape[1], fill_value=True)
        eer_where = np.full(er_EERs.shape[1], fill_value=True)

        # filter (reference, generator)-pairs according to script configuration
        for index, pair in enumerate(er_pairs):
            reference, generator = pair.split(":")

            include = True
            if args.which_dataset == "original":
                include = generator in get_wavefake_original_generators()
            elif args.which_dataset == "extended":
                include = generator in get_wavefake_extended_generators()

            if not args.include_JSUT and reference == "jsut":
                include = False

            acc_where[index] = include
            eer_where[index] = include

        # compute mean over available (reference, generator)-pairs
        acc_mean = np.nanmean(er_ACCs, axis=1, where=acc_where)
        eer_mean = np.nanmean(er_EERs, axis=1, where=eer_where)

        print(f"{'Acc. max':<16} " f"{'Acc. μ ±σ':<16} " f"{'EER min':<16} " f"{'EER μ ±σ':<16}")

        # now print max, mean, std over the available seeds
        print(
            f"{ff(np.max(acc_mean), p=True):<16} "
            f"{ff(np.mean(acc_mean), p=True) + ' ±' + ff(np.std(acc_mean), p=True):<16} "
            f"{ff(np.min(eer_mean)):<16} "
            f"{ff(np.mean(eer_mean)) + ' ±' + ff(np.std(eer_mean)):<16} "
        )
        print("\n----------------------------------------------------------------")
