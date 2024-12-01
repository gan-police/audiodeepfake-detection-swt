from typing import Optional, Tuple

import numpy as np
import pytest
import torch

import swtaudiofakedetect.dataset_transform as dtf
from swtaudiofakedetect.utils import inverse_permutation


class TestDatasetTransform:
    @pytest.mark.parametrize("shape", [(1024,), (4, 256)])
    @pytest.mark.parametrize("reshape", [(1, 1024), (1, 2, 512)])
    def test_reshape(self, shape: Tuple[int, ...], reshape: Tuple[int, ...]) -> None:
        test_data_np = np.random.random(shape)
        test_data_pt = torch.from_numpy(test_data_np)

        transformer = dtf.Reshape(reshape)

        result_np = transformer(test_data_np)
        result_pt = transformer(test_data_pt)

        assert result_np.shape == tuple(result_pt.shape)
        assert torch.allclose(torch.from_numpy(result_np), result_pt)

    @pytest.mark.parametrize("shape", [(1024,), (4, 256)])
    @pytest.mark.parametrize("reshape", [(1, 1024), (1, 2, 512)])
    def test_reshape_batch(self, shape: Tuple[int, ...], reshape: Tuple[int, ...]) -> None:
        test_data_np = np.random.random((4, *shape))
        test_data_pt = torch.from_numpy(test_data_np)

        transformer = dtf.Reshape(reshape, batch_mode=True)

        result_np = transformer(test_data_np)
        result_pt = transformer(test_data_pt)

        assert result_np.shape == tuple(result_pt.shape) == (4, *reshape)
        assert torch.allclose(torch.from_numpy(result_np), result_pt)

    @pytest.mark.parametrize("shape", [(8, 1024), (8, 4, 256)])
    def test_permute(self, shape: Tuple[int, ...]) -> None:
        test_data_np = np.random.random(shape)
        test_data_pt = torch.from_numpy(test_data_np)

        dims = tuple(np.roll(np.arange(len(shape)), 1))
        transformer = dtf.Permute(dims)

        result_np = transformer(test_data_np)
        result_pt = transformer(test_data_pt)

        assert result_np.shape == tuple(result_pt.shape)
        assert torch.allclose(torch.from_numpy(result_np), result_pt)

    @pytest.mark.parametrize("shape", [(1024,), (1024, 1024)])
    @pytest.mark.parametrize("size", [64, 128])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_fixed_slice(self, shape: Tuple[int, ...], size: int, axis: int) -> None:
        if axis >= len(shape):
            return

        test_data_np = np.zeros(shape)
        test_data_pt = torch.zeros(shape)

        transformer = dtf.FixedSlice(size, axis)

        result_np = transformer(test_data_np)
        assert result_np.shape[axis] == size

        result_pt = transformer(test_data_pt)
        assert result_pt.shape[axis] == size

        assert result_np.shape == tuple(result_pt.shape)

    @pytest.mark.parametrize("shape", [(1024,), (1024, 1024)])
    @pytest.mark.parametrize("size", [64, 128])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_random_slice(self, shape: Tuple[int, ...], size: int, axis: int) -> None:
        if axis >= len(shape):
            return

        test_data_np = np.zeros(shape)
        test_data_pt = torch.zeros(shape)

        transformer = dtf.RandomSlice(size, axis)

        result_np = transformer(test_data_np)
        assert result_np.shape[axis] == size

        result_pt = transformer(test_data_pt)
        assert result_pt.shape[axis] == size

        assert result_np.shape == tuple(result_pt.shape)

    @pytest.mark.parametrize("shape", [(8192,), (1, 8192), (4, 8192)])
    @pytest.mark.parametrize("wavelet", ["haar", "db2", "db4", "sym3"])
    def test_swt(self, wavelet: str, shape: Tuple[int, ...]) -> None:
        test_data_np = np.random.rand(*shape).astype(np.float64)
        test_data_pt = torch.from_numpy(test_data_np)

        swt = dtf.CalculateSWT(wavelet, 13)

        result_np = swt(test_data_np)
        result_pt = swt(test_data_pt)

        assert result_np.shape == tuple(result_pt.shape)

        # relative and absolute tolerances for float64 taken from
        # https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close
        assert torch.allclose(torch.from_numpy(result_np), result_pt, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize("wavelet", ["haar", "db2", "db4", "sym3"])
    def test_wpt(self, wavelet: str) -> None:
        test_data_np = np.random.rand(8192).astype(np.float64)
        test_data_pt = torch.from_numpy(test_data_np)

        wpt = dtf.CalculateWPT(wavelet, 7)

        result_np = wpt(test_data_np)
        result_pt = wpt(test_data_pt)

        assert result_np.shape == tuple(result_pt.shape)

        # relative and absolute tolerances for float64 taken from
        # https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close
        assert torch.allclose(torch.from_numpy(result_np), result_pt, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize("shape", [(3, 128, 128), (14, 8192)])
    @pytest.mark.parametrize("axis", [None, (0,), (1,), (0, 1)])
    def test_normalization(self, shape: Tuple[int, ...], axis: Optional[Tuple[int, ...]]) -> None:
        test_data = torch.rand(shape) + 42.0

        pt_mean = torch.mean(test_data, dim=axis)
        pt_std = torch.std(test_data, dim=axis)

        if axis is not None and any([v > i for i, v in enumerate(axis)]):
            permute = tuple([*axis] + [i for i in range(len(shape)) if i not in axis])
            test_data_n1 = (torch.permute(test_data, permute) - pt_mean) / pt_std
            test_data_n1 = torch.permute(test_data_n1, inverse_permutation(permute))
        else:
            permute = None
            test_data_n1 = (test_data - pt_mean) / pt_std

        normalizer = dtf.Normalize(pt_mean, pt_std, permute)

        test_data_n2 = normalizer(test_data)

        assert torch.allclose(test_data_n1, test_data_n2)

    @pytest.mark.parametrize("shape", [(3, 128, 128), (14, 8192)])
    def test_abslog(self, shape: Tuple[int, ...]) -> None:
        test_data_np = np.random.random(shape)
        test_data_np.flat[0] = 0  # set first element to zero, to ensure that at least one zero-value
        test_data_pt = torch.from_numpy(test_data_np)

        transformer = dtf.AbsLog()

        result_np = transformer(test_data_np)
        result_pt = transformer(test_data_pt)

        assert torch.allclose(torch.from_numpy(result_np), result_pt)

    @pytest.mark.parametrize("shape", [(14, 224), (3, 14, 224)])
    @pytest.mark.parametrize("repeats", [2, 4])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_repeat_interleave(self, shape: Tuple[int, ...], repeats: int, axis: int) -> None:
        test_data_np = np.random.random(shape).astype(np.float64)
        test_data_pt = torch.from_numpy(test_data_np)

        transformer = dtf.RepeatInterleave(repeats, axis)

        result_length = shape[axis] * repeats

        result_np = transformer(test_data_np)
        assert result_np.shape[axis] == result_length

        result_pt = transformer(test_data_pt)
        assert result_pt.shape[axis] == result_length

        assert result_np.shape == tuple(result_pt.shape)
        assert torch.allclose(torch.from_numpy(result_np), result_pt)

    def test_transforms_on_cuda(self) -> None:
        if torch.cuda.is_available():
            result = np.random.random((65536,)).astype(np.float64)

            transforms = dtf.Compose(
                [
                    dtf.RandomSlice(16384),  # (16384,)
                    dtf.ToTensor(device="cuda"),
                    dtf.CalculateSWT("haar", 14),  # (15, 16384)
                    dtf.AbsLog(),
                ]
            )

            for transform in transforms:
                result = transform(result)

                if isinstance(result, torch.Tensor):
                    assert result.is_cuda

    def test_transforms_on_cuda_with_dataloader(self) -> None:
        if torch.cuda.is_available():
            from torch.utils.data import DataLoader

            from swtaudiofakedetect.dataset import WFLoaded

            dataset = WFLoaded(
                "datasets/loaded",
                transform=dtf.Compose(
                    [
                        dtf.RandomSlice(16384),  # (16384,)
                        dtf.ToTensor(device="cuda"),
                        dtf.CalculateSWT("haar", 14),  # (15, 16384)
                        dtf.AbsLog(),
                    ]
                ),
            )

            loader = DataLoader(dataset, batch_size=4)

            for data, _, _ in loader:
                assert data.is_cuda
