import argparse
import os
import sys
import traceback

from swtaudiofakedetect.configuration import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()

    task_pid = os.getpid()
    print(f"[{task_pid}]: CPU Affinity={os.sched_getaffinity(0)}", file=sys.stderr)
    print(f"[{task_pid}]: CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', default='')}", file=sys.stderr)

    try:
        cfg = Config(args.config)
        module, kwargs = cfg.resolve_task(args.index)
        cfg.start_task(module, kwargs)
    except Exception as e:
        print(f"[{task_pid}]: {traceback.format_exc()}", file=sys.stderr)
        exit(1)
