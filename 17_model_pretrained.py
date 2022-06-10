import torchvision

# train_data = torchvision.datasets.ImageNet("./ImageNet_dataset", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# print(vgg16_true)

dataset = torchvision.datasets.CIFAR10("./CIFAR_dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))

# print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)