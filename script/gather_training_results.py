import argparse
import os
import os.path as path

from swtaudiofakedetect.configuration import Config
from swtaudiofakedetect.evaluation_utils import *
from swtaudiofakedetect.plot_utils import plot_training_progress

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str)
    group.add_argument("--directory", type=str)
    args = parser.parse_args()

    combined_paths = set()

    if args.config:
        cfg = Config(args.config)
        combined_paths = cfg.get_combined_output_paths()
    else:
        combined_paths.add(args.directory)

    for combined_path in combined_paths:
        paths = [
            path.join(combined_path, d) for d in os.listdir(combined_path) if path.isdir(path.join(combined_path, d))
        ]

        # checkpoints
        cp_paths = [path.join(p, f) for p in paths for f in os.listdir(p) if "checkpoint" in f]
        combined = combine_training_progress(cp_paths, path.join(combined_path, "training_progress_combined.pt"))
        plot_training_progress(combined, path.join(combined_path, "training_progress_combined.png"))

        # confusion matrices
        cm_paths = [
            path.join(p, "confusion_matrix.pt") for p in paths if path.isfile(path.join(p, "confusion_matrix.pt"))
        ]
        combine_confusion_matrices(cm_paths, path.join(combined_path, "confusion_matrix_combined.pt"))
