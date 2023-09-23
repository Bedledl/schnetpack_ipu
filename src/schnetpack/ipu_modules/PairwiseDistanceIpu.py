from typing import Dict

import torch

import schnetpack.properties as properties


class PairwiseDistancesIPU(torch.nn.Module):
    """
    Compute pair-wise distances from indices provided by a neighbor list transform.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        R = inputs[properties.R]
        offsets = inputs[properties.offsets]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # the backward pass of both python indexing and index_select is not supported in poptorch
        pos_j = R[idx_j]
        pos_i = R[idx_i]

        Rij = pos_j - pos_i + offsets
        inputs["R_ij_norm"] = torch.linalg.norm(Rij, dim=1)
        return inputs
