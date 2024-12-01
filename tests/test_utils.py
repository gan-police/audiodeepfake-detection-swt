from math import isclose

import pytest
import torch
from pandas import read_csv

from swtaudiofakedetect.evaluation_utils import create_evaluation_df
from swtaudiofakedetect.training_utils import calculate_eer


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


class TestUtils:
    @staticmethod
    def test_trainer_calculate_eer() -> None:
        y_true = torch.tensor([0, 0, 1, 1])
        y_score = torch.tensor([0.2, 0.4, 0.6, 0.8])
        assert isclose(calculate_eer(y_true, y_score), 0)
        assert isclose(calculate_eer(y_true, torch.roll(y_score, 1)), 0.5)
        assert isclose(calculate_eer(y_true, torch.flip(y_score, (0,))), 1)

    @staticmethod
    def test_evaluation_filter_df(create_simple_dataset) -> None:
        dataset_df = read_csv(create_simple_dataset + "/dataset.csv", keep_default_na=False)

        # BASIC5000_0001, BASIC5000_0002, BASIC5000_0003 of both reals and fakes
        train_indices = [0, 2, 8, 9, 11, 17]
        train_df = dataset_df[dataset_df.index.isin(train_indices)]

        result_df = create_evaluation_df(train_df, dataset_df, max_per_class=5)
        result_indices = result_df.index.to_list()

        # check if correct indices were removed
        for index in train_indices:
            assert index not in result_indices

        # check if BASIC5000_0004 for both reals and fakes is still included
        assert 7 in result_indices and 14 in result_indices

        # check if maximum 5 samples per class are included
        assert len(result_df.index) == 20
        assert all(result_df.groupby(["reference", "generator"]).size() == 5)
        # BASIC5000_0009 for both reals and fakes should not be included
        assert 1 not in result_indices and 12 not in result_indices

        # check correct conversion to iloc indices for pytorch subset dataset
        iloc_indices = dataset_df.index.get_indexer(result_df.index)
        assert 7 in iloc_indices and 1 not in iloc_indices
