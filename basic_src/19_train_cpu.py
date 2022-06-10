import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("./CIFAR_dataset", train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./CIFAR_dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
print("train data size: {}".format(len(train_data)))
print("test data size: {}".format(len(test_data)))

xv = Xv()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(xv.parameters(), learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("../logs")

for i in range(epoch):
    print("--------------epoch {} starting---------------".format(i+1))

    # training steps
    xv.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = xv(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("batch {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # testing steps
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = xv(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum()/len(test_data)
    print("Loss in whole epoch: {}".format(total_test_loss))
    print("Total accuracy: {}".format(total_accuracy))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step += 1

    torch.save(xv, "./saved_model/xv_epoch{}.pth".format(i))
    print("model saved")

writer.close()