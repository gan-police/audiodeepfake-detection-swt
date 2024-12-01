import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Self, Tuple, Union

import h5py
import numpy as np
import pandas
import torch
from torch.utils.data import Dataset

from swtaudiofakedetect.dataset_transform import Transformer
from swtaudiofakedetect.dataset_utils import (
    extract_dataframe_from_hdf5,
    filter_dataframe,
    Generators,
    load_sample,
    References,
)


class WaveFakeItemMeta(NamedTuple):
    name: str
    fake: bool
    reference: str
    generator: str

    @classmethod
    def from_row(cls, row: pandas.Series) -> Self:
        return cls(row["name"], row["fake"], row["reference"], row["generator"])


class WaveFakeDataset(ABC, Dataset):  # type: ignore
    df: pandas.DataFrame
    transform: Optional[Transformer] = None

    def __len__(self) -> int:
        return len(self.df.index)

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, torch.Tensor], torch.Tensor, int]:
        pass

    def get_meta(self, index: int) -> WaveFakeItemMeta:
        return WaveFakeItemMeta.from_row(self.get_row(index))

    def get_row(self, index: int) -> pandas.Series:
        return self.df.iloc[index]

    def get_df(self) -> pandas.DataFrame:
        return self.df

    def get_df_reals(self) -> pandas.DataFrame:
        return self.df.loc[self.df["fake"] == False]

    def get_df_fakes(self) -> pandas.DataFrame:
        return self.df.loc[self.df["fake"] == True]

    def count_reals(self, indices: Optional[List[int]] = None) -> int:
        if indices is not None:
            return sum(self.df.loc[self.df.index[indices], "fake"] == False)
        else:
            return len(self.get_df_reals().index)

    def count_fakes(self, indices: Optional[List[int]] = None) -> int:
        if indices is not None:
            return sum(self.df.loc[self.df.index[indices], "fake"] == True)
        else:
            return len(self.get_df_fakes().index)

    @abstractmethod
    def get_array_size(self) -> int:
        pass

    def get_dataframe_size(self) -> int:
        return self.df.memory_usage(deep=True).sum()


class WFSimple(WaveFakeDataset):
    """
    The WaveFake-Simple dataset loads the samples from the disk when requested.
    This dataset does not return SWT-transformed samples.

    Pros:
     - very lightweight on memory
    Cons:
     - TBD
    """

    def __init__(
        self,
        dataset_dir: str,
        target_sample_length: int,
        load_sample_rate: int = 22050,
        load_sample_offset: float = 0,
        load_sample_duration: Optional[float] = None,
        random_sample_slice: bool = False,
        references: Union[str, Iterable[str], None] = None,
        generators: Union[str, Iterable[str], None] = None,
        transform: Optional[Transformer] = None,
    ):
        self.csv_path = os.path.join(dataset_dir, "dataset.csv")
        self.df = pandas.read_csv(self.csv_path, keep_default_na=False)
        self.df = filter_dataframe(
            self.df,
            References(references) if references is not None else None,
            Generators(generators) if generators is not None else None,
        )

        self.target_sample_length = target_sample_length
        self.load_sample_rate = load_sample_rate
        self.load_sample_offset = load_sample_offset
        self.load_sample_duration = load_sample_duration
        self.random_sample_slice = random_sample_slice

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, torch.Tensor], torch.Tensor, int]:
        row: pandas.Series = self.get_row(index)

        # load the sample from the disk
        data = load_sample(
            row["path"],
            target_length=self.target_sample_length,
            load_sample_rate=self.load_sample_rate,
            load_sample_offset=self.load_sample_offset,
            load_sample_duration=self.load_sample_duration,
            random_sample_slice=self.random_sample_slice,
        )

        if row["fake"]:
            if self.transform is not None:
                return self.transform(data), torch.tensor([0, 1], dtype=torch.float64), index
            else:
                return data, torch.tensor([0, 1], dtype=torch.float64), index
        else:
            if self.transform is not None:
                return self.transform(data), torch.tensor([1, 0], dtype=torch.float64), index
            else:
                return data, torch.tensor([1, 0], dtype=torch.float64), index

    def get_array_size(self) -> int:
        return 0


class WFLoaded(WaveFakeDataset):
    """The WaveFake-Loaded dataset stores the audio files in memory. Therefore, it requires a preceding
    generation step, where all samples are loaded (librosa) in the desired shape and stored in one large array.
    This dataset does not return SWT-transformed samples.

    Pros:
     - avoids disk I/O when requesting a sample
    Cons:
     - requires dataset generation script before use
     - requires fair amount of main memory
    """

    def __init__(
        self,
        dataset_dir: str,
        references: Union[str, Iterable[str], None] = None,
        generators: Union[str, Iterable[str], None] = None,
        transform: Optional[Transformer] = None,
    ):
        self.h5_path = os.path.join(dataset_dir, "dataset.hdf5")
        self.h5_ub_size: int = 0
        self.data: np.ndarray

        with h5py.File(self.h5_path, "r") as h5:
            self.h5_ub_size = h5.userblock_size

        self.df = extract_dataframe_from_hdf5(self.h5_path, self.h5_ub_size)
        self.df = filter_dataframe(
            self.df,
            References(references) if references is not None else None,
            Generators(generators) if generators is not None else None,
        )

        with h5py.File(self.h5_path, "r") as h5:
            self.data = np.empty((len(self.df),) + h5["data"].shape[1:], dtype=h5["data"].dtype)
            h5["data"].read_direct(self.data, source_sel=np.s_[self.df.index])
            assert len(self.data) == len(self.df)

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, torch.Tensor], torch.Tensor, int]:
        row: pandas.Series = self.get_row(index)

        if row["fake"]:
            if self.transform is not None:
                return self.transform(self.data[index]), torch.tensor([0, 1], dtype=torch.float64), index
            else:
                return self.data[index], torch.tensor([0, 1], dtype=torch.float64), index
        else:
            if self.transform is not None:
                return self.transform(self.data[index]), torch.tensor([1, 0], dtype=torch.float64), index
            else:
                return self.data[index], torch.tensor([1, 0], dtype=torch.float64), index

    def get_array_size(self) -> int:
        return self.data.nbytes


class WFTransformed(WaveFakeDataset):
    """
    The WaveFake-Transformed dataset loads SWT-transformed samples from an HDF5-file into memory.

    Pros:
     - avoids disk I/O when requesting a sample
     - due to the preprocessing, no further action on the data is required before feeding it into the model
    Cons:
     - very expensive on memory
    """

    def __init__(
        self,
        dataset_dir: str,
        references: Union[str, Iterable[str], None] = None,
        generators: Union[str, Iterable[str], None] = None,
        transform: Optional[Transformer] = None,
    ):
        self.h5_path = os.path.join(dataset_dir, "dataset.hdf5")
        self.h5_ub_size: int = 0
        self.data: np.ndarray

        with h5py.File(self.h5_path, "r") as h5:
            self.h5_ub_size = h5.userblock_size

        self.df = extract_dataframe_from_hdf5(self.h5_path, self.h5_ub_size)
        self.df = filter_dataframe(
            self.df,
            References(references) if references is not None else None,
            Generators(generators) if generators is not None else None,
        )

        with h5py.File(self.h5_path, "r") as h5:
            self.data = np.empty((len(self.df),) + h5["data"].shape[1:], dtype=h5["data"].dtype)
            h5["data"].read_direct(self.data, source_sel=np.s_[self.df.index])
            assert len(self.data) == len(self.df)

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, torch.Tensor], torch.Tensor, int]:
        row: pandas.Series = self.get_row(index)

        if row["fake"]:
            if self.transform is not None:
                return self.transform(self.data[index]), torch.tensor([0, 1], dtype=torch.float64), index
            else:
                return self.data[index], torch.tensor([0, 1], dtype=torch.float64), index
        else:
            if self.transform is not None:
                return self.transform(self.data[index]), torch.tensor([1, 0], dtype=torch.float64), index
            else:
                return self.data[index], torch.tensor([1, 0], dtype=torch.float64), index

    def get_array_size(self) -> int:
        return self.data.nbytes


def initialize_dataset(dataset_type: str, dataset_dir: str, **kwargs: Dict[str, Any]) -> WaveFakeDataset:
    match dataset_type:
        case "simple":
            return WFSimple(dataset_dir, **kwargs)
        case "loaded":
            return WFLoaded(dataset_dir, **kwargs)
        case "transformed":
            return WFTransformed(dataset_dir, **kwargs)
        case _:
            raise ValueError("unexpected dataset type")
