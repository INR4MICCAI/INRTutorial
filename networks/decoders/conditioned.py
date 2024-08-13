import torch
from torch import nn
from networks.layers import AbstractLayer, ReluLayer


class ConditionedMLP(nn.Module):
    def __init__(self, coord_size: int, feat_size: int, hidden_size: int, num_layers: int,
                 layer_class: AbstractLayer = ReluLayer, **kwargs):
        super(ConditionedMLP, self).__init__()

        a = [layer_class(coord_size + feat_size, hidden_size, **kwargs)]
        for i in range(num_layers - 1):
            a.append(layer_class(hidden_size, hidden_size, **kwargs))
        self.hid_layers = nn.ModuleList(a)
        self.hidden_size = hidden_size

    @property
    def out_size(self):
        return self.hidden_size

    def forward(self, coord: torch.Tensor, feat: torch.Tensor):
        x = torch.cat((coord, feat), dim=1)
        for layer in self.hid_layers:
            x = layer(x)
        return x


class ConditionedResidualMLP(nn.Module):
    def __init__(self, coord_size: int, feat_size: int, hidden_size: int, num_layers: int,
                 layer_class: AbstractLayer = ReluLayer, **kwargs):
        super(ConditionedResidualMLP, self).__init__()

        assert coord_size == feat_size
        a = [layer_class(coord_size, hidden_size, **kwargs)]
        for i in range(num_layers - 1):
            a.append(layer_class(hidden_size, hidden_size, **kwargs))
        self.hid_layers = nn.ModuleList(a)

    @property
    def out_size(self):
        return self.hidden_size

    def forward(self, coord: torch.Tensor, feat: torch.Tensor):
        x = coord + feat
        prev_x = x
        for layer in self.hid_layers:
            x = layer(x) + prev_x
            prev_x = x
        return x
