import abc
from typing import Any, Union, Tuple, List, Iterable

import torch
from torch import nn
import numpy as np


class AbstractPosEncoder(nn.Module):
    def __init__(self, coord_size: int, **kwargs):
        super().__init__()
        self._in_size = coord_size

    @property
    def in_size(self):
        return self._in_size

    @property
    @abc.abstractmethod
    def out_size(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__} (in_size: {self.in_size}, out_size: {self.out_size})"


class IdentityPosEncoder(AbstractPosEncoder):
    def __init__(self, coord_size: int, **kwargs):
        super(IdentityPosEncoder, self).__init__(coord_size, **kwargs)

    @property
    def out_size(self):
        return self._in_size

    def forward(self, coords):
        return coords


# class NeRFPosEncoder(AbstractPosEncoder):
#     '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_frequencies = kwargs.get("num_frequencies")
#         assert isinstance(self.num_frequencies, tuple)
#         assert len(self.num_frequencies) == self.in_features
#
#         self.out_dim = self.in_features + 2 * np.sum(self.num_frequencies)
#
#     def forward(self, coords):
#         coords = coords.view(coords.shape[0], self.in_features)
#
#         coords_pos_enc = coords
#         for j, dim_freqs in enumerate(self.num_frequencies):
#             for i in range(dim_freqs):
#                 c = coords[..., j]
#
#                 sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
#                 cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
#
#                 coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
#
#         return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


class NeRFPosEncoder(AbstractPosEncoder):
    """ Vectorized version of the positional encoder from the official NeRF paper [Mildenhall et al. 2020].
     Original implementation found above. """
    def __init__(self, coord_size: int, freq_num: Union[int, Tuple[int], List[int]], freq_scale: float = 1.0, **kwargs):
        super(NeRFPosEncoder, self).__init__(coord_size, **kwargs)
        if isinstance(freq_num, int):
            freq_num = [freq_num] * coord_size
        elif isinstance(freq_num, (list, tuple)):
            assert len(freq_num) == coord_size
        self.freq_num = freq_num
        self.freq_scale = freq_scale
        # Pre-initialize the frequency components. Each coordinate may have a different number of frequencies.
        # TODO: Add this to GPU ahead of time? How to nicely ask the user for argument?
        self.exp_i_pi = torch.cat([2**torch.arange(i, dtype=torch.float32)[None] * self.freq_scale * np.pi
                                   for i in self.freq_num],
                                  dim=1)

    @property
    def out_size(self):
        return 2 * sum(self.freq_num)

    def forward(self, coords):
        coords_ = torch.cat([torch.tile(coords[..., j:j+1], (1, n)) for j, n in enumerate(self.freq_num)], dim=-1)
        exp_i_pi = torch.tile(self.exp_i_pi.to(coords.device), (coords_.shape[0], 1))
        prod = exp_i_pi * coords_
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


class FourierFeatPosEncoder(AbstractPosEncoder):
    """ Positional encoder from Fourite Features [Tancik et al. 2020]
     Implementation based on https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb """
    def __init__(self, coord_size: int, freq_num: int, freq_scale: float = 1.0, **kwargs):
        super(FourierFeatPosEncoder, self).__init__(coord_size, **kwargs)
        assert isinstance(freq_num, int)
        self.freq_num = freq_num
        self.freq_scale = freq_scale
        # TODO: Add this to GPU ahead of time? How to nicely ask the user for argument?
        self.B_gauss = torch.normal(0.0, 1.0, size=(coord_size, self.freq_num), requires_grad=False) * self.freq_scale
        self.B_gauss_pi = 2. * np.pi * self.B_gauss

    @property
    def out_size(self):
        return 2 * self.freq_num

    def get_extra_state(self) -> Any:
        """ Required to store gaussian array into network state dict.
        Otherwise the positional encoder will not be the same when you load a saved checkpoint. """
        return {"B_gauss_pi": self.B_gauss_pi}

    def set_extra_state(self, state: Any):
        """ Required to store gaussian array into network state dict.
        Otherwise the positional encoder will not be the same when you load a saved checkpoint. """
        self.B_gauss_pi = state["B_gauss_pi"]

    def forward(self, coords):
        prod = coords @ self.B_gauss_pi.to(coords.device)
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out
