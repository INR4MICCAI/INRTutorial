import abc

import torch
from torch import nn


class AbstractLayer(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(AbstractLayer, self).__init__()
        self._in_size = in_size
        self._out_size = out_size

    @property
    def in_size(self):
        return self._in_size

    @property
    def out_size(self):
        return self._out_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReLULayer(AbstractLayer):
    def __init__(self, in_size: int, out_size: int, **kwargs):
        super(ReLULayer, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        return x


class SIRENLayer(AbstractLayer):
    """
        Implicit Neural Representations with Periodic Activation Functions [Sitzmann et al. 2020]
        Implementation partly based on SIREN colab notebook:
        https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """
    def __init__(self, in_size, out_size, siren_factor=30., **kwargs):
        super(SIRENLayer, self).__init__(in_size, out_size, **kwargs)
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.siren_factor = siren_factor
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.siren_factor * x)
        return x


class WIRELayer(AbstractLayer):
    """
        WIRE: Wavelet Implicit Neural Representations [Saragadam et al. 2023]
        Implicit representation with Gabor nonlinearity
        Implementation based of https://github.com/vishwa91/wire
    """

    def __init__(self, in_size, out_size, wire_omega: float = 10.0, wire_sigma: float = 40.0, **kwargs):
        super(WIRELayer, self).__init__(in_size, out_size, **kwargs)
        self.omega_0 = wire_omega  # Freq
        self.scale_0 = wire_sigma
        self.freqs = nn.Linear(in_size, out_size)
        self.scale = nn.Linear(in_size, out_size)

    def forward(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        x = torch.cos(omega) * torch.exp(-(scale * scale))
        return x
