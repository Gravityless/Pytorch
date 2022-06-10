import torch
import torchvision
from PIL import Image
from model import *

img_path = "./dataset/dog.png"
img = Image.open(img_path).convert('RGB')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),
     torchvision.transforms.ToTensor()])

img = transform(img)
img = torch.reshape(img, (1, 3, 32, 32)).cuda()

model = torch.load("./saved_model/xv_epoch9.pth")
# model = torch.load("./saved_model/xv_epoch9.pth", map_location=torch.device("cpu"))
print(model)

model.eval()
with torch.no_grad():
    output = model(img)

print(output)
print(output.argmax(1))
