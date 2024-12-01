from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pywt
import torch
from ptwt.packets import WaveletPacket as pt_WaveletPacket
from ptwt.stationary_transform import swt as pt_swt

from swtaudiofakedetect.utils import inverse_permutation

Transformer = Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]]


class TransformerBase(ABC):
    @abstractmethod
    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()


class TransformerWithMode(TransformerBase):
    def __init__(self, *args, **kwargs) -> None:
        super(TransformerWithMode, self).__init__(*args, **kwargs)
        self._eval = False

    @abstractmethod
    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()

    def train_mode(self) -> None:
        self._eval = False

    def eval_mode(self) -> None:
        self._eval = True

    def is_eval(self) -> bool:
        return self._eval


class Compose(TransformerBase):
    """Compose multiple transforms into one callable class"""

    def __init__(self, transforms: List[Transformer]):
        self.transforms = transforms

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self) -> Iterator[Transformer]:
        for transform in self.transforms:
            yield transform


class ComposeWithMode(Compose):
    def set_train_mode(self) -> None:
        for transform in self.transforms:
            if isinstance(transform, TransformerWithMode):
                transform.train_mode()

    def set_eval_mode(self) -> None:
        for transform in self.transforms:
            if isinstance(transform, TransformerWithMode):
                transform.eval_mode()


class NoTransform(TransformerBase):
    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return sample


class ToTensor(TransformerBase):
    """Convert an input sample to a PyTorch tensor (dtype=float)"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(sample, np.ndarray):
            return torch.from_numpy(sample).to(dtype=torch.float32, device=self.device)
        else:
            return sample.to(dtype=torch.float32, device=self.device)


class Reshape(TransformerBase):
    """Reshape to a given shape"""

    def __init__(self, shape: Tuple[int, ...], batch_mode: bool = False):
        if batch_mode:
            self.shape = (-1, *shape)
        else:
            self.shape = shape

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            return np.reshape(sample, self.shape)
        elif isinstance(sample, torch.Tensor):
            return torch.reshape(sample, self.shape)
        else:
            raise ValueError("unexpected input type")


class Permute(TransformerBase):
    """Permute dimensions"""

    def __init__(self, dims: Tuple[int, ...]):
        self.dims = dims

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            return np.transpose(sample, self.dims)
        elif isinstance(sample, torch.Tensor):
            return torch.permute(sample, self.dims)
        else:
            raise ValueError("unexpected input type")


class Expand(TransformerBase):
    """Expand to one additional dimension by repeating an input N times"""

    def __init__(self, repeat_count: int):
        self.repeat_count = repeat_count

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            return np.tile(sample, (self.repeat_count, *[1 for _ in range(len(sample.shape))]))
        elif isinstance(sample, torch.Tensor):
            return sample.expand(self.repeat_count, *sample.shape)
        else:
            raise ValueError("unexpected input type")


class FixedSlice(TransformerBase):
    """Return the [0, size]-slice along the given dimension of an input, retaining other dimensions"""

    def __init__(self, size: int, axis: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size = size
        self.axis = axis

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            indices = np.arange(0, self.size)
            return np.take(sample, indices, axis=self.axis)
        elif isinstance(sample, torch.Tensor):
            indices = torch.arange(0, self.size, device=sample.device)
            return torch.index_select(sample, self.axis, indices)
        else:
            raise ValueError("unexpected input type")


class RandomSlice(TransformerWithMode):
    """Return a random fixed-size slice along the given dimension of an input, retaining other dimensions"""

    def __init__(self, size: int, axis: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size = size
        self.axis = axis

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval():
            if isinstance(sample, np.ndarray):
                indices = np.arange(0, self.size)
                return np.take(sample, indices, axis=self.axis)
            elif isinstance(sample, torch.Tensor):
                indices = torch.arange(0, self.size, device=sample.device)
                return torch.index_select(sample, self.axis, indices)
        else:
            length: int = sample.shape[self.axis]
            if length <= self.size:
                return sample
            else:
                index: int = np.random.randint(length - self.size)
                if isinstance(sample, np.ndarray):
                    indices = np.arange(index, index + self.size)
                    return np.take(sample, indices, axis=self.axis)
                elif isinstance(sample, torch.Tensor):
                    indices = torch.arange(index, index + self.size, device=sample.device)
                    return torch.index_select(sample, self.axis, indices)
        raise ValueError("unexpected input type")


class FixedCrop(TransformerWithMode):
    """Crop the given input of shape [..., H, W]"""

    def __init__(self, size: Tuple[int, int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size = size

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return sample[..., 0 : self.size[0], 0 : self.size[1]]


class CalculateSWT(TransformerBase):
    """Calculate the Stationary Wavelet Transform for some input"""

    def __init__(self, wavelet: str, levels: int, axis: int = -1) -> None:
        self.wavelet = pywt.Wavelet(wavelet)
        self.levels = levels
        self.axis = axis

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            # returns [cAn, cDn, ..., cD2, cD1] due to trim_approx=True
            coeffs = pywt.swt(sample, wavelet=self.wavelet, level=self.levels, axis=self.axis, trim_approx=True)
            # stack the coefficients vectors into one matrix
            return np.stack(coeffs)
        elif isinstance(sample, torch.Tensor):
            # returns [cAn, cDn, ..., cD2, cD1]
            coeffs = pt_swt(sample, wavelet=self.wavelet, level=self.levels, axis=self.axis)
            # stack the coefficients vectors into one matrix
            return torch.stack(coeffs)
        else:
            raise ValueError("unexpected input type")


class CalculateWPT(TransformerBase):
    """Calculate the Wavelet Packet Transform for some input"""

    def __init__(self, wavelet: str, levels: int, axis: int = -1) -> None:
        self.wavelet = pywt.Wavelet(wavelet)
        self.levels = levels
        self.axis = axis

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            wp_tree = pywt.WaveletPacket(
                sample, wavelet=self.wavelet, mode="reflect", maxlevel=self.levels, axis=self.axis
            )
            wp_nodes = wp_tree.get_level(self.levels, order="freq")
            wp_list: List[np.ndarray] = []

            for node in wp_nodes:
                wp_list.append(node.data)

            return np.stack(wp_list)
        elif isinstance(sample, torch.Tensor):
            wp_tree = pt_WaveletPacket(
                sample, wavelet=self.wavelet, mode="reflect", maxlevel=self.levels, axis=self.axis
            )

            wp_nodes = wp_tree.get_level(self.levels)
            wp_list: List[torch.Tensor] = []

            for node in wp_nodes:
                wp_list.append(wp_tree[node])

            return torch.stack(wp_list).squeeze()
        else:
            raise ValueError("unexpected input type")


class Normalize(TransformerBase):
    """Normalize input sample by subtracting the mean and dividing by the standard deviation"""

    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        permute: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
    ):
        self._mean, self._std, self._permute, self._device = mean, std, permute, device

    def do_permute(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.permute(sample, self._permute) if self._permute else sample

    def undo_permute(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.permute(sample, inverse_permutation(self._permute)) if self._permute else sample

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            return sample  # do not normalize

        sample = self.do_permute(sample)

        # subtract mean
        sample -= self._mean

        # divide by standard deviation
        sample /= self._std

        return self.undo_permute(sample)

    def set_mean_and_std(self, mean: torch.Tensor, std: torch.Tensor):
        self._mean, self._std = mean.to(device=self._device), std.to(device=self._device)

    def set_permute(self, permute: Tuple[int, ...]):
        self._permute = permute


class AbsLog(TransformerBase):
    """Take the logarithm of the absolute values of the input sample"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            return np.log10(np.abs(sample) + 1e-12)
        elif isinstance(sample, torch.Tensor):
            return torch.log10(torch.add(torch.abs(sample), 1e-12))
        else:
            raise ValueError("unexpected input type")


class RepeatInterleave(TransformerBase):
    """Repeat elements along an axis"""

    def __init__(self, repeats: int, axis: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.repeats = repeats
        self.axis = axis

    def __call__(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sample, np.ndarray):
            return np.repeat(sample, self.repeats, self.axis)
        elif isinstance(sample, torch.Tensor):
            return torch.repeat_interleave(sample, self.repeats, self.axis)
        else:
            raise ValueError("unexpected input type")
