import abc

import torch
from torch import nn
import math


class AbstractLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(AbstractLayer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
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


class ReluLayer(AbstractLayer):
    def __init__(self, in_size: int, out_size: int, bias: bool = True, **kwargs):
        super(ReluLayer, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class SineLayer(AbstractLayer):
    """ TODO: cite SIREN paper and github. """
    def __init__(self, in_size, out_size, siren_factor=30., **kwargs):
        super(SineLayer, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)
        self.weight_init()
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.siren_factor = siren_factor

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.siren_factor * x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def weight_init(self):
        with torch.no_grad():
            num_input = self.linear.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            self.linear.weight.uniform_(-math.sqrt(6 / num_input) / self.siren_factor, math.sqrt(6 / num_input) / self.siren_factor)


class ComplexWIRELayer(AbstractLayer):
    """
        TODO: cite WIRE paper and github.
        Implicit representation with complex Gabor nonlinearity
    """

    def __init__(self, in_size: int, out_size: int, bias: bool = True, wire_omega: float = 10.0, wire_sigma: float = 40.0, **kwargs):
        super(ComplexWIRELayer, self).__init__(in_size, out_size, **kwargs)
        self.omega_0 = wire_omega
        self.scale_0 = wire_sigma
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1))
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1))
        self.linear = nn.Linear(in_size, out_size, bias=bias, dtype=torch.cfloat)

    def forward(self, x):
        lin = self.linear(x)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        out = torch.exp(1j * omega - scale.abs().square())
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class WIRELayer(AbstractLayer):
    """
        TODO: cite WIRE paper and github.
        Implicit representation with Gabor nonlinearity
    """

    def __init__(self, in_size, out_size, bias=True, wire_omega: float = 10.0, wire_sigma: float = 40.0, **kwargs):
        super(WIRELayer, self).__init__(in_size, out_size, **kwargs)
        self.omega_0 = wire_omega  # Freq
        self.scale_0 = wire_sigma
        self.freqs = nn.Linear(in_size, out_size, bias=bias)
        self.scale = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        x = torch.cos(omega) * torch.exp(-(scale * scale))
        if self.dropout is not None:
            x = self.dropout(x)
        return x
