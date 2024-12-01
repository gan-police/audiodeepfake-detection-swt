import argparse
import os
import sys
import traceback

from swtaudiofakedetect.configuration import Config
from swtaudiofakedetect.evaluation_utils import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    task_pid = os.getpid()
    print(f"[{task_pid}]: CPU Affinity={os.sched_getaffinity(0)}", file=sys.stderr)
    print(f"[{task_pid}]: CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', default='')}", file=sys.stderr)

    try:
        cfg = Config(args.config)
        module, kwargs = cfg.resolve_task(args.index)

        # set model directory to output directory
        kwargs["model_dir"] = kwargs["output_dir"]
        # set model name if provided
        if args.model is not None:
            kwargs["model"] = args.model

        evaluate_model(**kwargs)
    except Exception as e:
        print(f"[{task_pid}]: {traceback.format_exc()}", file=sys.stderr)
        exit(1)
