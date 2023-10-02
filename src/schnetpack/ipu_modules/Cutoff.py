import math
import torch


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff: torch.Tensor):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, input: torch.Tensor):
        return 0.5 * (torch.cos(input * math.pi / self.cutoff) + 1.0)
