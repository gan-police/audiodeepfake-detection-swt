import os
from io import BytesIO
from math import ceil
from typing import Callable, Iterable, Iterator, List, NamedTuple, Optional, Self, Set, Union

import numpy as np
import pandas
from librosa import load

AVAILABLE_REFERENCES = ["jsut", "ljspeech"]
AVAILABLE_GENERATORS = [
    "melgan",
    "parallel_wavegan",
    "multi_band_melgan",
    "full_band_melgan",
    "hifiGAN",
    "waveglow",
    "avocodo",
    "bigvgan",
    "lbigvgan",
]

MAP_REFERENCE_SETS = {"jsut": "jsut_ver1.1/basic5000/wav", "ljspeech": "LJSpeech-1.1/wavs"}

MAP_GENERATOR_SETS = {
    "melgan": "WaveFake",
    "parallel_wavegan": "WaveFake",
    "multi_band_melgan": "WaveFake",
    "full_band_melgan": "WaveFake",
    "hifiGAN": "WaveFake",
    "waveglow": "WaveFake",
    "avocodo": "WaveFake-Extension",
    "bigvgan": "WaveFake-Extension",
    "lbigvgan": "WaveFake-Extension",
}

MAP_GENERATOR_NAMES = {
    "melgan": "MelGAN",
    "parallel_wavegan": "Parallel WaveGAN",
    "multi_band_melgan": "Multi-Band MelGAN",
    "full_band_melgan": "Full-Band MelGAN",
    "hifiGAN": "HiFi-GAN",
    "waveglow": "WaveGlow",
    "avocodo": "Avocodo",
    "bigvgan": "BigVGAN",
    "lbigvgan": "Large BigVGAN",
}


class SampleCSV(NamedTuple):
    name: str
    path: str
    reference: str
    generator: Optional[str]
    fake: bool


class Sample(NamedTuple):
    path: str
    reference: str
    generator: Optional[str]

    def to_sample_csv(self) -> SampleCSV:
        return SampleCSV(
            os.path.basename(self.path), self.path, self.reference, self.generator, self.generator is not None
        )


def files_iterator(downloads_dir: str) -> Iterator[Sample]:
    downloads_dir = os.path.abspath(downloads_dir)

    for reference in AVAILABLE_REFERENCES:
        dir_path: str = os.path.join(downloads_dir, MAP_REFERENCE_SETS[reference])
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".wav"):
                    yield Sample(os.path.join(dir_path, file_name), reference, None)

        for generator in AVAILABLE_GENERATORS:
            dir_path: str = os.path.join(downloads_dir, MAP_GENERATOR_SETS[generator], f"{reference}_{generator}")
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".wav"):
                        yield Sample(os.path.join(dir_path, file_name), reference, generator)


def files_batch_iterator(downloads_dir: str, batch_size: int) -> Iterator[List[Sample]]:
    batch: List[Sample] = [None] * batch_size  # type: ignore
    batch_idx = 0

    for sample in files_iterator(downloads_dir):
        batch[batch_idx] = sample
        batch_idx += 1

        if batch_idx == batch_size:
            yield batch
            batch_idx = 0

    if batch_idx > 0:
        yield batch[:batch_idx]


def load_sample(
    file_path: str,
    target_length: int,  # in samples
    load_sample_rate: int,
    load_sample_offset: float = 0,
    load_sample_duration: Optional[float] = None,
    random_sample_slice: bool = False,
) -> np.ndarray:
    data, _ = load(
        file_path, sr=load_sample_rate, offset=load_sample_offset, duration=load_sample_duration, dtype=np.float64
    )

    if len(data) == 0:
        # reload without offset
        data, _ = load(file_path, sr=load_sample_rate, duration=load_sample_duration, dtype=np.float64)

    if len(data) < target_length:
        # extend data array with zeros
        padl = (target_length - len(data)) // 2
        padr = ceil((target_length - len(data)) / 2)
        assert len(data) + padl + padr == target_length
        data = np.pad(data, (padl, padr), "constant", constant_values=0)

    if random_sample_slice and len(data) > target_length:
        idx = np.random.randint(len(data) - target_length)
        data = data[idx : idx + target_length]
    else:
        data = data[:target_length]

    assert len(data) == target_length

    # normalize data to [-1,1]
    data = data / np.abs(data).max()

    assert data.min() >= -1
    assert data.max() <= 1

    return data


def extract_dataframe_from_hdf5(file_path: str, userblock_size: int, chunk_size: int = 1024) -> pandas.DataFrame:
    with open(file_path, "br") as h5b:
        bio = BytesIO(h5b.read(userblock_size))

        # find first null character and truncate bytes buffer
        chunk = bytearray(chunk_size)
        counter = 0
        while bio.readinto(chunk):
            found = chunk.find(0)
            if found != -1:
                counter += found
                break
            else:
                counter += chunk_size

        bio.seek(0)
        bio.truncate(counter)

        return pandas.read_csv(bio, keep_default_na=False)


def is_valid_reference(reference: str) -> bool:
    return reference in AVAILABLE_REFERENCES


def is_valid_generator(generator: str) -> bool:
    return generator in AVAILABLE_GENERATORS


def validate_references(references: List[str]) -> bool:
    return all(is_valid_reference(x) for x in references)


def validate_generators(generators: List[str]) -> bool:
    return all(is_valid_generator(x) for x in generators)


class Builder:
    def __init__(self, validator: Callable[[str], bool], initial: Union[str, Iterable[str], None] = None) -> None:
        self._validator: Callable[[str], bool] = validator
        self._set: Set[str] = set()

        if isinstance(initial, str):
            for item in initial.split(","):
                item = item.strip()
                if not item:
                    continue
                elif not self._validator(item):
                    raise ValueError(f"item '{item}' is invalid")
                else:
                    self._set.add(item)

        elif initial is not None:
            for item in initial:
                item = item.strip()
                if not item:
                    continue
                elif not self._validator(item):
                    raise ValueError(f"item '{item}' is invalid")
                else:
                    self._set.add(item)

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self) -> Iterator[str]:
        for item in self._set:
            yield item

    def __contains__(self, item: str) -> bool:
        return item in self._set

    def list(self) -> List[str]:
        return list(self._set)

    def include(self, item: str) -> Self:
        if not self._validator(item):
            raise ValueError(f"item '{item}' is invalid")

        self._set.add(item)
        return self

    def exclude(self, item: str) -> Self:
        if not self._validator(item):
            raise ValueError(f"item '{item}' is invalid")

        self._set.remove(item)
        return self


class References(Builder):
    def __init__(self, initial: Union[str, Iterable[str], None] = None) -> None:
        super().__init__(is_valid_reference, initial)

    def all(self) -> Self:
        self._set = set(AVAILABLE_REFERENCES)
        return self


class Generators(Builder):
    def __init__(self, initial: Union[str, Iterable[str], None] = None) -> None:
        super().__init__(is_valid_generator, initial)

    def all(self) -> Self:
        self._set = set(AVAILABLE_GENERATORS)
        return self


def get_wavefake_original_generators() -> Generators:
    return Generators(["melgan", "parallel_wavegan", "multi_band_melgan", "full_band_melgan", "hifiGAN", "waveglow"])


def get_wavefake_extended_generators() -> Generators:
    return Generators(
        [
            "melgan",
            "parallel_wavegan",
            "multi_band_melgan",
            "full_band_melgan",
            "hifiGAN",
            "waveglow",
            "avocodo",
            "bigvgan",
            "lbigvgan",
        ]
    )


def filter_dataframe(
    df: pandas.DataFrame, references: Optional[References] = None, generators: Optional[Generators] = None
) -> pandas.DataFrame:
    if references is not None:
        df = df.loc[df["reference"].isin(references.list())]
    if generators is not None:
        df = df.loc[df["generator"].isin(generators.list() + [""])]
    return df
