import torch


class ShiftedSoftplusIPU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
        self.ss_module = torch.nn.Softplus()

    def forward(self, x):
        y = self.ss_module(x)
        return y - self.shift
