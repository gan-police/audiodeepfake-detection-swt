import argparse
import os

from yaml import dump

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["Wide16Basic", "Wide24Basic", "Wide19Bottle", "Wide32Bottle", "WptBasic", "WptBottle"],
    )
    parser.add_argument("--output_dir", type=str, default="out")
    args = parser.parse_args()

    if not os.path.exists(os.path.abspath(args.output_dir)):
        os.makedirs(os.path.abspath(args.output_dir))

    base = {
        "dataset_type": "loaded",
        "dataset_dir": "/p/scratch/hai_fnetwlet/datasets/loaded_AB_65536",
        "dataset_kwargs": {
            "references": ["ljspeech"],
            "generators": ["full_band_melgan"],
        },
        "batch_size": 128,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "weighted_loss": False,
        "lr_scheduler": True,
        "model": args.model_name,
    }

    if "Wide" in args.model_name:
        base["stop_epoch"] = 40
        base["num_checkpoints"] = 1
        base["num_validations"] = 20
        base["lr_milestones"] = [10, 20]
    else:
        base["stop_epoch"] = 80
        base["num_checkpoints"] = 1
        base["num_validations"] = 40
        base["lr_milestones"] = [25, 50]

    wavelets = [
        "haar",
        "db2",
        "db3",
        "db4",
        "db5",
        "db6",
        "db7",
        "db8",
        "db9",
        "db10",
        "sym2",
        "sym3",
        "sym4",
        "sym5",
        "sym6",
        "sym7",
        "sym8",
        "sym9",
        "sym10",
        "coif2",
        "coif3",
        "coif4",
        "coif5",
        "coif6",
        "coif7",
        "coif8",
        "coif9",
        "coif10",
    ]

    main_module = "script.training.wide_x_single" if "Wide" in args.model_name else "script.training.wpt_x_single"

    for k, wavelet in enumerate(wavelets):
        tasks = []
        for i in range(1, 5):
            tasks.append(
                {
                    "main_module": main_module,
                    "output_dir": f"/p/project/hai_fnetwlet/models/{args.model_name}_FBM/{wavelet}/{i}",
                    "wavelet": wavelet,
                    "seed": i,
                }
            )

        with open(os.path.join(args.output_dir, f"{k + 1:03d}_{wavelet}.yaml"), "w") as yaml_file:
            dump(base | {"tasks": tasks}, yaml_file, default_flow_style=False)
