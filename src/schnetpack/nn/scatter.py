import torch
from torch_scatter import scatter_add

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
    return scatter_add(x, idx_i, dim_size=dim_size, dim=dim)
