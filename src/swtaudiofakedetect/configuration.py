import argparse
import importlib
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Set, Tuple

from yaml import load, Loader

from swtaudiofakedetect.utils import KWArgsAppend


def parse_args_to_kwargs(parser=argparse.ArgumentParser()) -> Dict[str, Any]:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_kwargs", nargs="*", default={}, action=KWArgsAppend)
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--stop_epoch", type=int, default=10)
    parser.add_argument("--num_validations", type=int, default=4)
    parser.add_argument("--num_checkpoints", type=int, default=0)
    return vars(parser.parse_args())


class Config:
    def __init__(self, config: str):
        if os.path.isfile(os.path.abspath(config)):
            with open(os.path.abspath(config), "r") as file:
                self.config = load(file, Loader=Loader)
        else:
            self.config = load(config, Loader=Loader)

        if "dataset_dir" not in self.config:
            raise ValueError("missing 'dataset_dir' string")
        elif not isinstance(self.config["dataset_dir"], str):
            raise ValueError("expected 'dataset_dir' to be a string")
        if "tasks" not in self.config:
            raise ValueError("missing 'tasks' list")
        elif not isinstance(self.config["tasks"], list):
            raise ValueError("expected 'tasks' to be a list")

        self.default = {
            "seed": 42,
            "batch_size": 64,
            "num_workers": 0,
            "wavelet": "haar",
            "stop_epoch": 10,
            "num_validations": 4,
            "num_checkpoints": 0,
        }

    def num_tasks(self) -> int:
        return len(self.config["tasks"])

    def get_task(self, index: int) -> Tuple[str, Dict[str, Any]]:
        module_name = ""
        main_kwargs = deepcopy(self.default)

        for k, v in self.config.items():
            if k != "tasks":
                main_kwargs[k] = v

        for k, v in self.config["tasks"][index].items():
            if k == "main_module":
                module_name = v
            else:
                main_kwargs[k] = v

        return module_name, main_kwargs

    def get_combined_output_paths(self) -> Set[str]:
        # get all out paths
        out_paths = []
        for task_idx in range(self.num_tasks()):
            _, kwargs = self.get_task(task_idx)
            out_paths.append(kwargs["output_dir"])

        # combine out paths to common parent paths
        combined_paths = set()

        if len(out_paths) > 0:
            combined_paths.add(out_paths[0])

            for new_path in out_paths:
                common_paths = []
                for combined_path in combined_paths:
                    common_paths.append((os.path.commonpath([combined_path, new_path]), combined_path))

                commonality_found = False
                for common_path, combined_path in common_paths:
                    # if common path is parent directory of new_path
                    if common_path.rstrip("/").count("/") == new_path.rstrip("/").count("/") - 1:
                        combined_paths.remove(combined_path)
                        combined_paths.add(common_path)
                        commonality_found = True
                        break

                if not commonality_found:
                    combined_paths.add(new_path)

        return combined_paths

    def resolve_task(self, index: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if index is None:
            index = int(os.getenv("SLURM_PROCID", default="-1"))

        if index < 0 or index >= self.num_tasks():
            raise RuntimeError(f"bad task index = {index}")

        return self.get_task(index)

    @staticmethod
    def start_task(module: str, kwargs: dict) -> None:
        if module == "":
            raise RuntimeError(f"missing main module for task")

        mod = importlib.import_module(module)
        mod.main(**kwargs)
