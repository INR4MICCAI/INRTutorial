import torch
from torch import nn
from typing import Tuple


class SegmentationHead(nn.Module):
    def __init__(self, in_size, num_classes: int, bias: bool = True, **kwargs):
        super(SegmentationHead, self).__init__()
        self.seg_layer = nn.Linear(in_size, num_classes, bias=bias)

    def forward(self, x: torch.Tensor):
        out = self.seg_layer(x)
        out = nn.functional.softmax(out, dim=1)
        return out


class ReconstructionHead(nn.Module):
    def __init__(self, in_size, num_channels: int = 1, bias: bool = True, **kwargs):
        super(ReconstructionHead, self).__init__()
        self.out_layer = nn.Linear(in_size, num_channels, bias=bias)

    def forward(self, x: torch.Tensor):
        out = self.out_layer(x)
        out = torch.sigmoid(out)
        return out


class RegistrationHead(nn.Module):
    def __init__(self, in_size, num_coords: int = 2, bias: bool = True, **kwargs):
        super(RegistrationHead, self).__init__()
        self.out_layer = nn.Linear(in_size, num_coords, bias=bias)

    def forward(self, x: torch.Tensor):
        out = self.out_layer(x)
        out = torch.tanh(out)
        return out
