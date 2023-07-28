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
        idx_i_expanded = idx_i.unsqueeze(1).expand(idx_i.shape[0], 3)
        pos_i = torch.gather(R, 0, idx_i_expanded)
        idx_j_expanded = idx_j.unsqueeze(1).expand(idx_j.shape[0], 3)
        pos_j = torch.gather(R, 0, idx_j_expanded)

        Rij = pos_j - pos_i + offsets
        norm = Rij.pow(2).sum(-1)
        # because we didn't filter out the loops yet, we have some 0 values here in the backward pass
        norm = torch.sqrt(norm + 1e-8)
        inputs[properties.Rij_norm] = norm
        return inputs
