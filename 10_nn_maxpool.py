import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))

class Xv(nn.Module):
    def __init__(self):
        super(Xv, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

xv = Xv()
output = xv(input)
print(output)

dataset = torchvision.datasets.CIFAR10("./CIFAR_dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Xv(nn.Module):
    def __init__(self):
        super(Xv, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

xv = Xv()

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, target = data
    output = xv(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()