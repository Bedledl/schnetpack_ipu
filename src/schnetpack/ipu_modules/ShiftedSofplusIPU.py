import torch


class ShiftedSoftplusIPU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        u = torch.log(1 + torch.exp(x))
        return u - self.shift
