import abc

import torch
from torch import nn


class AbstractRepresentation(nn.Module):
    def __init__(self, num_subjects: int, feat_size: int):
        super(AbstractRepresentation, self).__init__()
        self.num_subjects = num_subjects
        self.feat_size = feat_size

    @property
    def out_size(self):
        return self.feat_size

    @abc.abstractmethod
    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError


class InstanceRepresentation(nn.Module):
    def __init__(self, num_subjects: int, feat_size: int, feat_std: float = 1.0):
        super(InstanceRepresentation, self).__init__(num_subjects, feat_size)
        self.representations = nn.Parameter(torch.normal(0.0, feat_std, (num_subjects, feat_size)), requires_grad=True)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        return self.representations[idx]
