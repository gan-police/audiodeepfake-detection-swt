# Audio Deepfake Detection using the Stationary Wavelet Transform

This repository contains the source code for the thesis "[Audio Deepfake Detection using the Stationary Wavelet Transform](https://doi.org/10.34734/FZJ-2024-04917)".


**University:** University of Bonn\
**Project Supervisor:** Dr. Moritz Wolter\
**Examiner:** Prof. Dr. Estela Suarez\
**Date:** June 2024 

Previous and related work includes "[Towards generalizing deep-audio-fake detection networks](https://openreview.net/pdf?id=RGewtLtvHz)", which in particular studies the Wavelet Packet Transforms for audio signals.

### Datasets

For authentic audio we use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (2017) and [JSUN](https://arxiv.org/pdf/1711.00354) (2017).

For the fake audio we use the [WaveFake](https://zenodo.org/records/5642694) dataset and its [extension](https://zenodo.org/records/10512541).\
These datasets include the following generator models:
- Full-Band MelGAN (2020)
- HiFi-GAN (2020)
- MelGAN (2019)
- Multi-Band MelGAN (2020)
- Parallel WaveGAN (2020)
- WaveGlow (2018)
- [Avocodo](https://arxiv.org/pdf/2206.13404) (2023)
- [BigVGAN](https://arxiv.org/pdf/2206.04658) (2023)

### System Configuration

One compute node of the JUWELS-Booster cluster at Jülich Supercomputing Centre (JSC).
Check the [hardware configuration](evaluations/system) for more details.


### Installation

`python -m pip install .`

### Job configuration

#### config.yaml - Synopsis

| Parameter          | Required | Type | Default | Scope      |
|--------------------|----------|------|---------|------------|
| tasks              | YES      | list |         | root       |
| seed               | NO       | int  | 42      | root, task |
| dataset_type       | YES      | str  |         | root, task |
| dataset_dir        | YES      | str  |         | root, task |
| dataset_kwargs     | NO       | map  |         | root, task |
| batch_size         | NO       | int  | 64      | root, task |
| num_workers        | NO       | int  | 0       | root, task |
| persistent_workers | NO       | bool | false   | root, task |
| wavelet            | NO       | str  | haar    | root, task |
| main_module        | YES      | str  |         | task       |
| output_dir         | YES      | str  |         | task       |
| checkpoint_file    | NO       | str  |         | task       |
| stop_epoch         | NO       | int  | 10      | root, task |
| num_validations    | NO       | int  | 4       | root, task |
| num_checkpoints    | NO       | int  | 0       | root, task |

All further parameters will also be passed to the main_module via kwargs.\
If a main_module does not support a given argument, it will raise a TypeError.

#### config.yaml - Example

<pre>dataset_type: simple
dataset_dir: ./datasets/simple/
dataset_kwargs:
  target_sample_length: 16834
  load_sample_offset: 0.25
  random_sample_slice: true
batch_size: 4
num_workers: 4
tasks:
  -
    main_module: src.models.wide_6_conv
    output_dir: ./out/wide_6_conv_unweighted/
    stop_epoch: 10
    num_checkpoints: 0
    num_validations: 10
  -
    main_module: src.models.wide_6_conv
    output_dir: ./out/wide_6_conv_even/
    stop_epoch: 10
    num_checkpoints: 0
    num_validations: 10
    custom_sampler: true
    custom_sampler_mode: even
</pre>

## Training and Evaluation

### Training a model

`sbatch --ntasks=4 --cpus-per-task=6 --threads-per-core=1 --time=01:30:00 script/run_training_and_gather.sh script/config/009_wide_x_single.yaml`

## Evaluating a model

`sbatch --time=00:20:00 script/run_devel.sh -O script/evaluate_model.py --model_dir /p/project/hai_fnetwlet/models/wide_32_FBM_haar --model_name Wide32 --model_checkpoint checkpoint_40.pt --train_wavelet haar --dataset_type loaded --dataset_dir /p/scratch/hai_fnetwlet/datasets/loaded_AB_65536 --dataset_kwargs references=ljspeech --output_dir /p/project/hai_fnetwlet/models/wide_32_FBM_haar/evaluation --num_workers 2`