import argparse
import dataclasses
import json
import os
from typing import List

from scipy.io import wavfile


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclasses.dataclass
class RealSampleDirectory:
    dataset_id: str
    path_to_wav: str


@dataclasses.dataclass
class FakeSampleDirectory:
    dataset_id: str
    reference_dataset_id: str
    generator_model_id: str
    path_to_wav: str


@dataclasses.dataclass
class SummarizedDirectory:
    directory: RealSampleDirectory | FakeSampleDirectory
    sample_count: int
    sample_rate: int
    sample_dtype: str
    samples_min: float | int
    samples_max: float | int

    def print(self):
        output = ""
        match type(self.directory).__name__:
            case "RealSampleDirectory":
                output += f"## REAL ## {self.directory.dataset_id} ## ({self.directory.path_to_wav})\n"
            case "FakeSampleDirectory":
                output += f"## FAKE ## {self.directory.generator_model_id} on {self.directory.reference_dataset_id} ## ({self.directory.path_to_wav})\n "

        output += f"samples={self.sample_count}, sample_rate={self.sample_rate}, sample_dtype={self.sample_dtype}\n"
        output += f"samples_min={self.samples_min}, samples_max={self.samples_max}\n\n"
        print(output)

    def store(self):
        name = "summary"

        match type(self.directory).__name__:
            case "RealSampleDirectory":
                name += f"_{self.directory.dataset_id}"
            case "FakeSampleDirectory":
                name += f"_{self.directory.dataset_id}_{self.directory.reference_dataset_id}_{self.directory.generator_model_id}"

        name += ".json"

        with open(os.path.join(args.OUTPUT_DIRECTORY, name), "w") as file:
            json.dump(self, file, ensure_ascii=False, cls=EnhancedJSONEncoder)


WAVEFAKE_DATASET_ID = "wavefake"
LJSPEECH_DATASET_ID = "ljspeech"
JSUT_DATASET_ID = "jsut"

# MelGAN
MELGAN_GENERATOR_ID = "melgan"
# Parallel WaveGAN
PWAVEGAN_GENERATOR_ID = "parallel_wavegan"
# Multi-Band MelGAN
MULTIMELGAN_GENERATOR_ID = "multi_band_melgan"
# Full-Band MelGAN
FULLMELGAN_GENERATOR_ID = "full_band_melgan"
# HiFi-GAN
HIFIGAN_GENERATOR_ID = "hifiGAN"
# WaveGlow
WAVEGLOW_GENERATOR_ID = "waveglow"

parser = argparse.ArgumentParser()
parser.add_argument("--DOWNLOADS_DIRECTORY", type=str)
parser.add_argument("--OUTPUT_DIRECTORY", type=str)
args = parser.parse_args()

directories: List[RealSampleDirectory | FakeSampleDirectory] = [
    RealSampleDirectory(
        dataset_id=LJSPEECH_DATASET_ID, path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "LJSpeech-1.1/wavs")
    ),
    RealSampleDirectory(
        dataset_id=JSUT_DATASET_ID, path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "jsut_ver1.1/basic5000/wav")
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=JSUT_DATASET_ID,
        generator_model_id=MULTIMELGAN_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/jsut_multi_band_melgan"),
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=JSUT_DATASET_ID,
        generator_model_id=PWAVEGAN_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/jsut_parallel_wavegan"),
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=LJSPEECH_DATASET_ID,
        generator_model_id=FULLMELGAN_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/ljspeech_full_band_melgan"),
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=LJSPEECH_DATASET_ID,
        generator_model_id=HIFIGAN_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/ljspeech_hifiGAN"),
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=LJSPEECH_DATASET_ID,
        generator_model_id=MELGAN_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/ljspeech_melgan"),
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=LJSPEECH_DATASET_ID,
        generator_model_id=MULTIMELGAN_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/ljspeech_multi_band_melgan"),
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=LJSPEECH_DATASET_ID,
        generator_model_id=PWAVEGAN_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/ljspeech_parallel_wavegan"),
    ),
    FakeSampleDirectory(
        dataset_id=WAVEFAKE_DATASET_ID,
        reference_dataset_id=LJSPEECH_DATASET_ID,
        generator_model_id=WAVEGLOW_GENERATOR_ID,
        path_to_wav=os.path.join(args.DOWNLOADS_DIRECTORY, "WaveFake/ljspeech_waveglow"),
    ),
]

# TODO: missing directories: "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech" and "ljspeech_melgan_large"

for i, directory in enumerate(directories):
    summary = SummarizedDirectory(directory, 0, 0, "", float("inf"), float("-inf"))

    for filename in os.listdir(directory.path_to_wav):
        try:
            sample_rate, data = wavfile.read(os.path.join(directory.path_to_wav, filename))
        except:
            print(f"failed to read {os.listdir(directory.path_to_wav)}")
            continue

        summary.sample_count += 1

        if summary.sample_rate == 0:
            summary.sample_rate = sample_rate
        elif summary.sample_rate != sample_rate:
            print(
                f"found sample with different sample rate: {os.path.join(directory.path_to_wav, filename)}, "
                f"expected {summary.sample_rate}, got {sample_rate}"
            )
        if summary.sample_dtype == "":
            summary.sample_dtype = str(data.dtype)
        elif summary.sample_dtype != str(data.dtype):
            print(
                f"found sample with different sample dtype: {os.path.join(directory.path_to_wav, filename)}, "
                f"expected {summary.sample_dtype}, got {str(data.dtype)}"
            )

        summary.samples_min = min(summary.samples_min, len(data))
        summary.samples_max = max(summary.samples_max, len(data))

    summary.store()
