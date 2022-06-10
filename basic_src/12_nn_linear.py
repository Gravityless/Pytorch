import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Xv(nn.Module):
    def __init__(self):
        super(Xv, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        return self.linear1(input)

dataset = torchvision.datasets.CIFAR10("./CIFAR_dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

xv = Xv()

# writer = SummaryWriter("./logs")
# step = 0
for data in dataloader:
    imgs, target = data
    output = xv(torch.flatten(imgs))

    # writer.add_images("input", imgs, step)
    # writer.add_images("output", output, step)
    # step += 1

# writer.close()