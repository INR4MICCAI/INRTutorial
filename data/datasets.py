from typing import Iterable, Dict, Any, Tuple, Optional, List, Union
import abc

import torch
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
import random
import numpy as np
from medmnist import ChestMNIST

from data.utils import min_max_normalize


class RandomPointsDataset(Dataset):
    def __init__(self, points_num: int):
        super(RandomPointsDataset, self).__init__()
        self.points_num = points_num
        sample_coords, saple_values, _ = self[0]
        self.coord_size = sample_coords.shape[-1]
        self.value_size = saple_values.shape[-1]

    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_image(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor],
                                            Union[Tuple[int, ...], List[int]]]:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        image, spatial_dims = self.load_image(idx)
        image = torch.tensor(image, dtype=torch.float32)
        # Create random sample of pixel indices
        point_indices = [torch.randint(0, i, (self.points_num,)) for i in spatial_dims]

        # Retrieve image values from selected indices
        point_values = image[tuple(point_indices)]

        # Convert point indices into normalized [-1, 1.0] coordinates
        point_coords = torch.stack(point_indices, dim=-1)
        spatial_dims = torch.tensor(spatial_dims)
        point_coords_norm = point_coords / (spatial_dims / 2) - 1

        # The subject index is also returned in case the user wants to use subject-wise learned latents
        return point_coords_norm, point_values, idx


class RandomPointsWithImageDataset(Dataset):
    def __init__(self, points_num: int):
        super(RandomPointsWithImageDataset, self).__init__()
        self.points_num = points_num

    @abc.abstractmethod
    def load_image(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor],
                                            Union[Tuple[int, ...], List[int]]]:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        image, spatial_dims = self.load_image(idx)
        image = torch.tensor(image, dtype=torch.float32)

        # Create random sample of pixel indices
        point_indices = [torch.randint(0, int(i), (self.points_num,)) for i in spatial_dims]

        # Retrieve image values from selected indices
        point_values = image[tuple(point_indices)]

        # Convert point indices into normalized [-1, 1.0] coordinates
        point_coords = torch.stack(point_indices, dim=-1)
        spatial_dims = torch.tensor(spatial_dims)
        point_coords_norm = point_coords / (spatial_dims / 2) - 1

        # The subject index is also returned in case the user wants to use subject-wise learned latents
        return image, point_coords_norm, point_values, idx


class MedMNISTDataset(RandomPointsDataset):
    def __init__(self, images, num_points: int = 1000):
        self.images = list(images)[0:1]
        super(MedMNISTDataset, self).__init__(num_points)

    def __len__(self):
        return len(self.images)

    def load_image(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        image, _ = self.images[idx]
        image = pil_to_tensor(image)
        image = min_max_normalize(image)
        spatial_dims = image.shape[1:]
        image = image.moveaxis(0, -1)
        return image, spatial_dims
