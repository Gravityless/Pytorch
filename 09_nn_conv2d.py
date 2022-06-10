import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Xv(nn.Module):
    def __init__(self):
        super(Xv, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)

xv = Xv()
# print(xv)

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = xv(imgs)
    output = torch.reshape(output, [-1, 3, 30, 30])

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()