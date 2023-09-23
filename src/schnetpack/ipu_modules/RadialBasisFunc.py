import torch
from torch import Tensor


class GaussianRBFIPU(torch.nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(self, n_rbf: int, cutoff: float, start: float = 0.0, trainable=False):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBFIPU, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        if trainable:
            self.offsets = torch.nn.Parameter(offset)
        else:
            self.register_buffer("offsets", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist[..., None] - self.offsets
        dist = torch.pow(dist, 2)
        y = torch.exp(self.coeff * dist)
        return y

