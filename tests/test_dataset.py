from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, random_split

from swtaudiofakedetect.dataset import WFLoaded, WFSimple, WFTransformed
from swtaudiofakedetect.dataset_normalization import WelfordEstimator
from swtaudiofakedetect.dataset_sampler import clean_generated_name, WaveFakeBatchSampler
from swtaudiofakedetect.dataset_utils import (
    extract_dataframe_from_hdf5,
    filter_dataframe,
    Generators,
    load_sample,
    References,
)
from swtaudiofakedetect.utils import seed_rngs


@pytest.fixture(scope="class")
def create_simple_dataset() -> str:
    from tempfile import TemporaryDirectory

    from script.generate_dataset import main

    tmp_dir = TemporaryDirectory()
    main(
        DOWNLOADS_DIRECTORY="downloads",
        OUTPUT_DIRECTORY=tmp_dir.name,
        DATASET_TYPE="simple",
        NUM_WORKERS=0,
        BATCH_SIZE=8,
    )

    yield tmp_dir.name
    tmp_dir.cleanup()


@pytest.fixture(scope="class")
def create_loaded_dataset() -> str:
    from tempfile import TemporaryDirectory

    from script.generate_dataset import main

    tmp_dir = TemporaryDirectory()
    main(
        DOWNLOADS_DIRECTORY="downloads",
        OUTPUT_DIRECTORY=tmp_dir.name,
        DATASET_TYPE="loaded",
        NUM_WORKERS=1,
        BATCH_SIZE=8,
        TARGET_SAMPLE_LENGTH=65536,
    )

    yield tmp_dir.name
    tmp_dir.cleanup()


@pytest.fixture(scope="class")
def create_transformed_dataset() -> str:
    from tempfile import TemporaryDirectory

    from script.generate_dataset import main

    tmp_dir = TemporaryDirectory()
    main(
        DOWNLOADS_DIRECTORY="downloads",
        OUTPUT_DIRECTORY=tmp_dir.name,
        DATASET_TYPE="transformed",
        NUM_WORKERS=1,
        BATCH_SIZE=8,
        SWT_LEVELS=10,
    )

    yield tmp_dir.name
    tmp_dir.cleanup()


class TestDataset:
    def test_load_sample(self) -> None:
        data = load_sample(
            "downloads/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav", target_length=2**14, load_sample_rate=22050
        )
        assert len(data) == 2**14

    def test_extract_df_from_h5(self, create_loaded_dataset) -> None:
        df = extract_dataframe_from_hdf5(create_loaded_dataset + "/dataset.hdf5", 8192)
        assert len(df) == 36

    def test_references_builder(self) -> None:
        builder = References()

        assert len(builder) == 0
        assert len(builder.include("jsut")) == 1
        assert builder.list()[0] == "jsut"
        assert len(builder.all()) == 2
        assert len(builder.exclude("jsut")) == 1

        with pytest.raises(ValueError):
            builder.include("bad?")

        with pytest.raises(ValueError):
            builder.exclude("bad?")

        builder = References("jsut")
        assert len(builder) == 1

        builder = References("jsut,")
        assert len(builder) == 1

        builder = References("jsut,ljspeech , jsut ")
        assert len(builder) == 2

        builder = References(["jsut", "ljspeech"])
        assert len(builder) == 2

        with pytest.raises(ValueError):
            References("bad?")

    def test_generators_builder(self) -> None:
        builder = Generators()

        assert len(builder) == 0
        assert len(builder.include("melgan")) == 1
        assert builder.list()[0] == "melgan"
        assert len(builder.all()) == 9
        assert len(builder.exclude("melgan")) == 8

        with pytest.raises(ValueError):
            builder.include("bad?")

        with pytest.raises(ValueError):
            builder.exclude("bad?")

    def test_filter_dataframe(self, create_loaded_dataset) -> None:
        # get dataframe
        df = extract_dataframe_from_hdf5(create_loaded_dataset + "/dataset.hdf5", 8192)
        assert len(df) == 36

        # filter by reference
        references = References().include("jsut")
        assert len(filter_dataframe(df.copy(deep=True), references=references)) == 18

        # filter by generator
        generators = Generators().include("multi_band_melgan")
        assert len(filter_dataframe(df.copy(deep=True), generators=generators)) == 27

        # filter by reference and generator
        generators = Generators().all().exclude("multi_band_melgan")
        assert len(filter_dataframe(df.copy(deep=True), references=references, generators=generators)) == 9

    @pytest.mark.parametrize("shape", [(100, 3, 128, 128), (100, 14, 8192)])
    @pytest.mark.parametrize("axis", [None, (0,), (0, 1), (0, 2)])
    @pytest.mark.parametrize("batch_size", [4, 8])
    def test_welford_estimator(self, shape: Tuple[int, ...], axis: Optional[Tuple[int, ...]], batch_size: int) -> None:
        test_data = np.random.randn(*shape).astype(np.float64) * 1e6

        welford = WelfordEstimator(shape=shape, axis=axis)

        batch_data = np.empty((batch_size, *shape[1:]), dtype=np.float64)
        batch_idx = 0
        for i in range(len(test_data)):
            batch_data[batch_idx] = test_data[i, :]
            batch_idx += 1

            if batch_idx == batch_size:
                welford.update(torch.from_numpy(batch_data))
                batch_idx = 0

        if batch_idx > 0:
            welford.update(torch.from_numpy(batch_data[:batch_idx]))

        welford_mean, welford_std = welford.finalize()

        np_mean = np.mean(test_data, axis=axis)
        np_std = np.std(test_data, axis=axis)

        assert np.allclose(np_mean, welford_mean)
        assert np.allclose(np_std, welford_std)

    def test_dataset_simple(self, create_simple_dataset) -> None:
        dataset = WFSimple(create_simple_dataset, target_sample_length=16384)
        assert len(dataset) == 36
        assert dataset.count_reals() == 18
        assert dataset.count_fakes() == 18

        # get item
        data, label, _ = dataset[0]

        assert data.shape == (16384,)
        assert label.shape == (2,)

    def test_dataset_loaded(self, create_loaded_dataset) -> None:
        dataset = WFLoaded(create_loaded_dataset)
        assert len(dataset) == 36
        assert dataset.count_reals() == 18
        assert dataset.count_fakes() == 18

        # get item
        data, label, _ = dataset[0]

        assert data.shape == (65536,)
        assert label.shape == (2,)

    def test_dataset_transformed(self, create_transformed_dataset) -> None:
        dataset = WFTransformed(create_transformed_dataset)
        assert len(dataset) == 36
        assert dataset.count_reals() == 18
        assert dataset.count_fakes() == 18

        # get item
        data, label, _ = dataset[0]

        assert data.shape == (11, 1024)
        assert label.shape == (2,)

    def test_batch_sampler_even(self, create_loaded_dataset) -> None:
        dataset = WFLoaded(create_loaded_dataset, references=References().include("ljspeech").list())

        sampler = WaveFakeBatchSampler(dataset.df, batch_size=5, mode="even")

        reals_count = 0
        fakes_count = 0

        for batch in sampler:
            for index in batch:
                row = dataset.df.loc[index]
                if row["fake"]:
                    fakes_count += 1
                else:
                    reals_count += 1

        assert reals_count == fakes_count

    def test_batch_sampler_fair(self, create_loaded_dataset) -> None:
        dataset = WFLoaded(create_loaded_dataset, references=References().include("ljspeech").list())

        sampler = WaveFakeBatchSampler(dataset.df, batch_size=4, mode="fair")

        reals_count = 0
        fakes_count = 0

        for batch in sampler:
            names = []
            for index in batch:
                row = dataset.df.loc[index]
                names.append(row["name"])
                if row["fake"]:
                    fakes_count += 1
                else:
                    reals_count += 1

            for x in range(len(names) // 2):
                assert any(
                    [
                        x != y and clean_generated_name(names[x]) == clean_generated_name(names[y])
                        for y in range(len(names))
                    ]
                )

        assert reals_count == fakes_count

    def test_batch_sampler_shuffle(self, create_loaded_dataset) -> None:
        seed_rngs()

        dataset = WFLoaded(create_loaded_dataset)

        sampler = WaveFakeBatchSampler(dataset.df, batch_size=3, shuffle=True)

        reals_count = 0
        fakes_count = 0
        first_index_1 = None
        first_index_2 = None

        for epoch in range(2):
            for i, batch in enumerate(sampler):
                for j, index in enumerate(batch):
                    if epoch == 0 and i == j == 0:
                        first_index_1 = index
                    elif epoch == 1 and i == j == 0:
                        first_index_2 = index

                    row = dataset.df.loc[index]
                    if row["fake"]:
                        fakes_count += 1
                    else:
                        reals_count += 1

        assert reals_count == fakes_count
        assert first_index_1 != first_index_2

    def test_batch_sampler_indices(self, create_loaded_dataset) -> None:
        dataset = WFLoaded(create_loaded_dataset)
        # indices [0, 8] and [18, 26] are real
        # indices [9, 17] and [27, 35] are fake

        indices = [4, 20, 10, 14, 30, 34]
        # indices 4, 20 are from real samples and will be silently ignored
        # indices 10, 14, 30, 34 are fake samples -> len(sampler) = 4 * 2 = 8

        sampler = WaveFakeBatchSampler(dataset.df, batch_size=3, indices=indices)

        assert len(sampler) == 8

        reals_count = 0
        fakes_count = 0

        for batch in sampler:
            for index in batch:
                row = dataset.df.loc[index]
                if row["fake"]:
                    assert any(x == row.iloc[0] for x in indices)
                    fakes_count += 1
                else:
                    reals_count += 1

        assert reals_count == fakes_count

    def test_batch_sampler_loader(self, create_loaded_dataset) -> None:
        seed_rngs()

        dataset = WFLoaded(create_loaded_dataset)

        subset1, subset2 = random_split(dataset, [0.5, 0.5])

        sampler1 = WaveFakeBatchSampler(dataset.df, batch_size=4, indices=subset1.indices)
        sampler2 = WaveFakeBatchSampler(dataset.df, batch_size=4, indices=subset2.indices)

        loader1 = DataLoader(dataset=dataset, batch_sampler=sampler1)
        loader2 = DataLoader(dataset=dataset, batch_sampler=sampler2)

        fakes_count = 0
        for data, labels, _ in loader1:
            fakes_count += labels[:, 1].sum().item()

        for data, labels, _ in loader2:
            fakes_count += labels[:, 1].sum().item()

        assert fakes_count == dataset.count_fakes()
