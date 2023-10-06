import math
import torch


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff: torch.Tensor):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, input: torch.Tensor):
        input_cut = 0.5 * (torch.cos(input * math.pi / self.cutoff) + 1.0)
        input_cut = input_cut *  (input < self.cutoff).to(torch.float32)
        return input_cut
