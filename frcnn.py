import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

# All code in this file was our own implementation


image_path = "data/images/bike.jpg"
image = Image.open(image_path)


model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    predictions = model(input_tensor)

boxes = predictions[0]['boxes']
labels = predictions[0]['labels']

fig, ax = plt.subplots(1)
ax.imshow(image)
for box, label in zip(boxes, labels):
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    ax.text(box[0], box[1], label, color='red')

plt.show()
