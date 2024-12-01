import argparse

from swtaudiofakedetect.evaluation_utils import evaluate_model
from swtaudiofakedetect.utils import KWArgsAppend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "Wide6l1dCNN",
            "Wide6l2dCNN",
            "Wide16Basic",
            "Wide24Basic",
            "Wide19Bottle",
            "Wide32Bottle",
            "WptBasic",
            "WptBottle",
            "DCNN",
        ],
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_checkpoint", type=str, required=True)
    parser.add_argument("--train_csv", type=str, default="train_dataset.csv")
    parser.add_argument("--train_norm_mean_std", type=str, default="train_mean_std.pt")
    parser.add_argument("--train_wavelet", type=str, default="haar")
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_kwargs", nargs="*", default={}, action=KWArgsAppend)
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    evaluate_model(
        args.model_name,
        args.model_dir,
        args.output_dir,
        stop_epoch=0,
        wavelet=args.train_wavelet,
        dataset_type=args.dataset_type,
        dataset_dir=args.dataset_dir,
        dataset_kwargs=args.dataset_kwargs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_checkpoint=args.eval_checkpoint,
        train_csv=args.train_csv,
        train_norm_mean_std=args.train_norm_mean_std,
    )
