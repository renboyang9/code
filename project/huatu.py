import torch
import torchvision
from PIL import Image
from torch import nn
from torch.version import cuda
import torchvision.models as models
from torchvision.transforms import transforms

model1 = torchvision.models.vgg16(pretrained=False)
model1.load_state_dict(torch.load("./save_model/vgg16.pth"))


model_weights =[]

conv_layers = []
model_children = list(model1.features.children())
counter = 0

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")


print(f"Total convolution layers: {counter}")
model = model1.cuda()
list(model1.children())
image = Image.open(str(r'E:\net-vlad-pytorch\Pitts\huatu.jpg'))
testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
image = testing_transforms(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.cuda()

outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(50, 60))
for i in range(len(processed)):
    a = fig.add_subplot(7, 5, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')