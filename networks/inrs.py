import math
import torch
from torch import nn
from networks.layers import ReLULayer, SIRENLayer, WIRELayer


class ReLUMLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 **kwargs):
        super().__init__()

        a = [ReLULayer(in_size, hidden_size, **kwargs)]
        for i in range(num_layers - 1):
            a.append(ReLULayer(hidden_size, hidden_size, **kwargs))
        a.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(a)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class SIRENMLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 siren_factor: float = 30.,
                 **kwargs):
        super().__init__()

        a = [SIRENLayer(in_size, hidden_size, siren_factor=siren_factor, **kwargs)]
        for i in range(num_layers - 1):
            a.append(SIRENLayer(hidden_size, hidden_size, siren_factor=siren_factor, **kwargs))
        a.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(a)
        self.initialize_weights(siren_factor)

    def initialize_weights(self, omega: float):
        """ See SIREN paper supplement Sec. 1.5 for discussion """
        old_weights = self.layers[1].linear.weight.clone()
        with torch.no_grad():
            # First layer initialization
            num_input = self.layers[0].linear.weight.size(-1)
            self.layers[0].linear.weight.uniform_(-1 / num_input, 1 / num_input)
            # Subsequent layer initialization uses based on omega parameter
            for layer in self.layers[1:-1]:
                num_input = layer.linear.weight.size(-1)
                layer.linear.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            # Final linear layer also uses initialization based on omega parameter
            num_input = self.layers[-1].weight.size(-1)
            self.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)

        # Verify that weights did indeed change
        new_weights = self.layers[1].linear.weight
        assert (old_weights - new_weights).abs().sum() > 0.0

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class WIRENMLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 wire_omega: float = 10.0,
                 **kwargs):
        super().__init__()

        a = [WIRELayer(in_size, hidden_size, wire_omega=wire_omega, **kwargs)]
        for i in range(num_layers - 1):
            a.append(WIRELayer(hidden_size, hidden_size, wire_omega=wire_omega, **kwargs))
        a.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(a)
        self.initialize_weights(wire_omega)

    def initialize_weights(self, omega: float):
        old_weights = self.layers[1].freqs.weight.clone()
        with torch.no_grad():
            # First layer initialization
            num_input = self.layers[0].freqs.weight.size(-1)
            self.layers[0].freqs.weight.uniform_(-1 / num_input, 1 / num_input)
            self.layers[0].scale.weight.uniform_(-1 / num_input, 1 / num_input)
            # Subsequent layer initialization based on omega parameter
            for layer in self.layers[1:-1]:
                num_input = layer.freqs.weight.size(-1)
                layer.freqs.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
                layer.scale.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            # Final linear layer also uses initialization based on omega parameter
            num_input = self.layers[-1].weight.size(-1)
            self.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            self.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)

        # Verify that weights did indeed change
        new_weights = self.layers[1].freqs.weight
        assert (old_weights - new_weights).abs().sum() > 0.0

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
