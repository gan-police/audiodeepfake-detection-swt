import argparse
import os
import random
import re
from math import floor
from time import perf_counter_ns
from typing import AnyStr, Callable, Tuple, Union

import numpy as np
import torch


def seed_rngs(random_seed: int = 42) -> None:
    """Manually seed all random number generators for reproducibility"""

    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def print_bytes(value: int) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.02f}GB"
    elif value >= 1e6:
        return f"{value / 1e6:.02f}MB"
    elif value >= 1e3:
        return f"{value / 1e3:.02f}KB"

    return f"{value}B"


def print_duration(value: int) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.03f}s"
    elif value >= 1e6:
        return f"{value / 1e6:.03f}ms"
    elif value >= 1e3:
        return f"{value / 1e3:.03f}µs"

    return f"{value}ns"


def format_float(v: float, p: bool = False) -> str:
    if not p:
        return "{:.03f}".format(round(v, 3))
    else:
        return "{:.02f}%".format(round(v * 100, 2))


def inverse_permutation(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    inv = [0] * len(perm)
    for i in range(len(perm)):
        inv[perm[i]] = i
    return tuple(inv)


def get_conv2d_output_shape(
    shape: Union[int, Tuple[int, int]],
    kernel: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    x, y = shape if isinstance(shape, tuple) else (shape, shape)
    kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    return floor((x + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1), floor(
        (y + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
    )


def get_maxpool2d_output_shape(
    shape: Union[int, Tuple[int, int]],
    kernel: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int], None] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    x, y = shape if isinstance(shape, tuple) else (shape, shape)
    kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
    stride = (stride, stride) if isinstance(stride, int) else stride if stride is not None else kernel
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    return floor((x + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1), floor(
        (y + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
    )


class KWArgsAppend(argparse.Action):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rx_bool: re.Pattern[AnyStr] = re.compile(r"^ *([Tt](rue)?|[Ff](alse)?) *$")
        self.rx_float: re.Pattern[AnyStr] = re.compile(r"^ *[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)? *$")
        self.rx_int: re.Pattern[AnyStr] = re.compile(r"^ *[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+) *$")

    def __call__(self, parser, args, values, option_string=None) -> None:
        try:
            d = dict(map(lambda x: x.split("="), values))
            # parse values to appropriate types
            for k, v in d.items():
                if bool(self.rx_bool.match(v)):
                    d[k] = "T" in v or "t" in v
                elif bool(self.rx_int.match(v)):
                    d[k] = int(v, 0)  # use base 0 for prefix-guessing behavior
                elif bool(self.rx_float.match(v)):
                    d[k] = float(v)
                # else we assume a string value
        except ValueError:
            raise argparse.ArgumentError(self, f"argument requires key=value format")
        setattr(args, self.dest, getattr(args, self.dest) | d)


class BenchmarkContext:
    def __init__(self, message: str, logger: Callable[[str], None] = print) -> None:
        self.message = message
        self.logger = logger

    def __enter__(self) -> None:
        self.start = perf_counter_ns()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.logger(self.message + print_duration(perf_counter_ns() - self.start))
