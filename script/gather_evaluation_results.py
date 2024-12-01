import argparse
import os
import os.path as path

import torch

from swtaudiofakedetect.configuration import Config
from swtaudiofakedetect.evaluation_utils import *

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

        # evaluation results
        er_paths = [
            path.join(p, "evaluation_results.pt") for p in paths if path.isfile(path.join(p, "evaluation_results.pt"))
        ]
        er_results = [torch.load(p, map_location=torch.device("cpu")) for p in er_paths]
        combined = combine_evaluation_results(er_results, path.join(combined_path, "evaluation_results_combined.pt"))
        print_evaluation_results(combined, path.join(combined_path, "evaluation_results_combined.txt"))
