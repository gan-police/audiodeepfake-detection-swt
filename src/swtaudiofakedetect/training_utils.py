import os
from typing import Optional

import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

from swtaudiofakedetect.logger import Logger
from swtaudiofakedetect.utils import seed_rngs


# setting up a reproducible training environment
def setup(seed: int, output_dir: str, log_file: str = "output.log") -> Logger:
    # ensure reproducibility
    seed_rngs(seed)
    if not __debug__:
        torch.use_deterministic_algorithms(True)

    # ensure output directory exists
    if not os.path.exists(os.path.abspath(output_dir)):
        os.makedirs(os.path.abspath(output_dir))

    logger = Logger(os.path.join(output_dir, log_file))
    return logger


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model(model_name: str, checkpoint_path: Optional[str] = None, *args, **kwargs) -> torch.nn.Module:
    module: torch.nn.Module
    match model_name:
        case "Wide6l1dCNN":
            from swtaudiofakedetect.models.wide_6_1d_conv import Wide6l1dCNN

            module = Wide6l1dCNN(*args, **kwargs)
        case "Wide6l2dCNN":
            from swtaudiofakedetect.models.wide_6_2d_conv import Wide6l2dCNN

            module = Wide6l2dCNN(*args, **kwargs)
        case "Wide16Basic":
            from swtaudiofakedetect.models.wide_basic import Wide16Basic

            module = Wide16Basic(*args, **kwargs)
        case "Wide24Basic":
            from swtaudiofakedetect.models.wide_basic import Wide24Basic

            module = Wide24Basic(*args, **kwargs)
        case "Wide19Bottle":
            from swtaudiofakedetect.models.wide_bottle import Wide19Bottle

            module = Wide19Bottle(*args, **kwargs)
        case "Wide32Bottle":
            from swtaudiofakedetect.models.wide_bottle import Wide32Bottle

            module = Wide32Bottle(*args, **kwargs)
        case "WptBasic":
            from swtaudiofakedetect.models.wpt_basic import WptBasic

            module = WptBasic(*args, **kwargs)
        case "WptBottle":
            from swtaudiofakedetect.models.wpt_bottle import WptBottle

            module = WptBottle(*args, **kwargs)
        case "DCNN":
            from swtaudiofakedetect.models.dcnn import DCNN

            module = DCNN(*args, **kwargs)
        case _:
            raise ValueError(f"'{model_name}' is not a valid model")

    if checkpoint_path is not None:
        checkpoint_path = torch.load(checkpoint_path, map_location=get_device())
        module.load_state_dict(checkpoint_path["model_state_dict"])

    return module


def calculate_eer(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Calculate the equal error rate (EER), a single measure for the overall accuracy of binary classifiers. It is
    defined as the point on the receiver operating characteristic (ROC) curve, where the false acceptance rate and the
    false rejection rate are equal. Values equal to zero indicate no wrong predictions, whereas values equal to one
    indicate only incorrect predictions. A value of 0.5 can be interpreted as guessing.

    :param y_true: Ground truth labels, values either 0 or 1 (binary)
    :param y_score: Probabilities for the positive class, values between 0 and 1
    """

    fpr, tpr, thresholds = roc_curve(y_true.numpy(), y_score.numpy())
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer
