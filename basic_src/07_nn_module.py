import torch
from torch import nn

class Xv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

xv = Xv()
input = torch.tensor(1.0)
output = xv(input)
print(output)
