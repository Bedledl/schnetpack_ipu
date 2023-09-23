"""
With the knowledge from the minimal_dis.py example we can build a neighbor distance module that
can be differentiated on the IPU with autograd.grad
Also this module does no support batching
"""
from typing import Dict

import torch
import poptorch
from schnetpack import properties
from schnetpack.transform import Transform

from poptorch_geometric.ops import knn_graph


class KNNNeighborTransform(Transform):
    """
    Returns the k-nearest Neighbors.
    This class does not inherit from the Schnetpack Neighbor Transformations
    because we can process whole batches with this implementation,
    because of poptorch_geometric.knn_graph
    also we assume that the idx_i value is given as input and is usually in the form of
    torch.arange(n_atoms * n_molecules).repeat_interleave(k)
    for example for k = 3, n_atoms = 5, n_molecules = 2
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    if we know that the batch size says constant throughout the use of this Module, we can use the
    n_atoms and n_molecules parameter to create a buffer with constant idx_i input.
    """
    def __init__(self, k, n_replicas, n_atoms, cutoff_shell=2., always_update=True):
        super(KNNNeighborTransform, self).__init__()
        self.k = int(k)
        self.n_atoms = n_atoms
        self.n_replicas = n_replicas
        self.cutoff_shell = cutoff_shell
        self.always_update = True
        self.register_buffer("previous_positions", torch.full((n_replicas * n_atoms, 3), float('nan')))
        self.register_buffer("previous_idx_j", torch.zeros(n_replicas * n_atoms * k, dtype=torch.int32))

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        def calc_nl(positions: torch.Tensor, n_replicas, n_atoms, k) -> torch.Tensor:
            idx_j_batches = []

            offset = 0
            for batch_pos in torch.chunk(positions, n_replicas):
                x_expanded = batch_pos.expand(batch_pos.size(0), *batch_pos.shape)
                y_expanded = batch_pos.reshape(batch_pos.size(0), 1, batch_pos.size(1))

                diff = x_expanded - y_expanded
                norm = torch.linalg.norm(diff, dim=2)

                dist, col = torch.topk(norm,
                                       k=k + 1,  # we need k + 1 because topk inclues loops
                                       dim=0,
                                       largest=False)
                # somehow when using this distance values the gradients after the filter network are zero
                # but they are the same values as we get with the Distance transform

                # this removes all the loops
                col = col.reshape(-1, k + 1)[:, 1:].reshape(-1)
                idx_j_batches.append(col + offset)

                offset += n_atoms

            idx_j = torch.cat(idx_j_batches)

            return idx_j, positions

        if inputs.get(properties.idx_i, None) is None:
            raise ValueError("Input dictionary nor self.idx_i do not contain the idx_i value. "
                             "This can lead to errors throughout the NN.")

        if inputs.get(properties.offsets, None) is None:
            raise ValueError("Input dictionary nor self.offsets do not contain the offsets value. "
                             "This can lead to errors throughout the NN.")

        # TODO batching! without a real batching implemnetation we mix atoms from different molecules
        positions = inputs[properties.position]

        idx_j, positions = calc_nl(positions, self.n_replicas, self.n_atoms, self.k)

        #self.previous_idx_j.copy_(idx_j)
        #self.previous_positions.copy_(positions)

        inputs[properties.idx_j] = idx_j

        return inputs
