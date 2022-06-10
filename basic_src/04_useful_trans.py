from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")
img = Image.open("../seperate_dataset/train/ants_image/0013035.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_totensor(trans_resize(img))
writer.add_image("Resize", img_resize, 0)

# Compose
trans_resize_1 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_1, trans_totensor])
img_resize_1 = trans_compose(img)
writer.add_image("Resize", img_resize_1, 1)

# RandomCrop
trans_random_1 = transforms.RandomCrop(512)
trans_compose_1 = transforms.Compose([trans_random_1, trans_totensor])
for i in range(10):
    img_crop = trans_compose_1(img)
    writer.add_image("RandomCrop", img_crop, i)

trans_random_2 = transforms.RandomCrop((300, 600))
trans_compose_2 = transforms.Compose([trans_random_2, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop_2", img_crop, i)

writer.close()
