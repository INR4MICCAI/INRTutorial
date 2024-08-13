from typing import Optional
import torch


def min_max_normalize(image: torch.Tensor, min: Optional[float] = None, max: Optional[float] = None):
    if min is None:
        min = image.min()
    if max is None:
        max = image.max()
    image_norm = (image - min) / (max - min)
    return image_norm
