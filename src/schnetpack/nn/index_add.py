import torch
from torch import nn

__all__ = ["index_add"]


def index_add(x: torch.Tensor,
              dim: int,
              index: torch.Tensor,
              source: torch.Tensor) -> torch.Tensor:
    if len(index) > source.size(dim):
        raise ValueError("Index tensor must not be longer than source tensor.")

    if dim > 1:
        raise NotImplementedError

    if dim == 0:
        for i in range(len(index)):
            x[index[i]] += source[i]
    elif dim == 1:
        for j in range(x.size(1)):
            for i in range(len(index)):
                x[j, index[i]] += source[j, i]

    return x
