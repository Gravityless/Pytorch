import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# method 1
torch.save(vgg16, "./saved_model/vgg16_method1.pth")
model = torch.load("./saved_model/vgg16_method1.pth")

# method 2
torch.save(vgg16.state_dict(), "./saved_model/vgg16_method2.pth")
model = torchvision.models.vgg16(pretrained=False)
model.load_state_dict(torch.load("./saved_model/vgg16_method2.pth"))

print(model)
