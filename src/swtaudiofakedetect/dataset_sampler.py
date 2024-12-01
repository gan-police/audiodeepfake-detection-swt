from abc import ABC
from typing import Iterator, List, Literal, Optional

import numpy as np
from pandas import DataFrame
from torch.utils.data import Sampler


def clean_generated_name(name: str) -> str:
    return name.replace("_gen", "").replace("erated", "")


class WaveFakeBatchSampler(Sampler[List[int]], ABC):
    """Samples batches from the WaveFakeDataset with an even distribution of real and fake samples.
    This sampler guarantees that one full iteration contains exactly many real samples as fake samples. Additionally,
    for an even batch size, this sampler also guarantees that every batch contains exactly as many real samples as fake
    samples. Moreover, in "fair" mode, this sampler returns the matching real sample for every fake sample within every
    batch. However, if the batch size is not even, a batch may contain one fake sample with no matching real sample.
    Note, that this sampler will ignore provided indices of real samples and always use all the real samples contained
    in the dataset instance. It is also important to note, that this sampler may sample real samples multiple times
    in one full iteration, if the number of real samples is less than the number of fake samples.
    """

    def __init__(
        self,
        data_frame: DataFrame,
        batch_size: int,
        shuffle: bool = False,
        mode: Literal["even", "fair"] = "even",
        indices: Optional[List[int]] = None,
    ):
        super().__init__()

        reals = data_frame.loc[data_frame["fake"] == False]
        fakes = data_frame.loc[data_frame["fake"] == True]

        if indices is not None:
            fakes = fakes[fakes.index.isin(indices)]

        self.real_indices = np.zeros(len(fakes.index), dtype=int)
        self.fake_indices = np.array(fakes.index, dtype=int)

        for i in range(len(self.fake_indices)):
            if mode == "even":
                # just select some real sample for every fake sample
                self.real_indices[i] = reals.iloc[i % len(reals.index)].iloc[0]
            elif mode == "fair":
                # select exactly the suiting real sample for the given fake sample
                fake_row = fakes.iloc[i]
                sample_name = clean_generated_name(fake_row["name"])
                real_matches = reals[reals["name"] == sample_name]
                if len(real_matches) > 0:
                    self.real_indices[i] = real_matches.iloc[0].iloc[0]
                else:
                    # inform about the missing suitable real sample
                    print(f"could not find real sample for {fake_row['name']} ({sample_name})")
                    # select an arbitrary real sample and ignore mode="fair" for this index
                    self.real_indices[i] = reals.iloc[i % len(reals.index)].iloc[0]
            else:
                raise ValueError("mode should either be 'even' or 'fair'")

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.fake_indices) + len(self.real_indices)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            permutation = np.random.permutation(len(self.fake_indices))
            self.real_indices = self.real_indices[permutation]
            self.fake_indices = self.fake_indices[permutation]

        fake_index = 0
        real_index = 0
        num_batches = (len(self.fake_indices) * 2) // self.batch_size
        for batch_step in range(num_batches):
            batch_indices: List[int] = []

            for sample_step in range(self.batch_size):
                if sample_step % 2 == batch_step % 2:
                    batch_indices.append(self.fake_indices[fake_index])
                    fake_index += 1
                else:
                    batch_indices.append(self.real_indices[real_index])
                    real_index += 1

            yield batch_indices

        if fake_index < len(self.fake_indices) or real_index < len(self.real_indices):
            source_switch = 0 if fake_index < real_index else 1
            batch_indices: List[int] = []

            for sample_step in range(len(self.fake_indices) - fake_index + len(self.real_indices) - real_index):
                if sample_step % 2 == source_switch:
                    batch_indices.append(self.fake_indices[fake_index])
                    fake_index += 1
                else:
                    batch_indices.append(self.real_indices[real_index])
                    real_index += 1

            yield batch_indices
