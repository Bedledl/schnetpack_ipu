import torch
from torch import nn

__all__ = ["scatter_add"]


def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.
    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce
    Returns:
        reduced input
    """
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(tuple(shape), dtype=x.dtype, device=x.device)

    if(len(x.shape) == 3):
        expanded_idx = idx_i.unsqueeze(1).repeat(1, shape[2]).unsqueeze(1).repeat(1,shape[1],1)
    elif (len(x.shape) == 2):
        expanded_idx = idx_i.unsqueeze(1)
    else:
        raise NotImplementedError

    expanded_idx = expanded_idx.repeat(1, shape[1])

    result = tmp.scatter_add_(dim, expanded_idx, x)
    return result
