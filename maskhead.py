import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from torchvision.models.detection.image_list import ImageList
from skimage.measure import find_contours


class FasterRCNNWithMask(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNNWithMask, self).__init__()
        self.fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.fasterrcnn.roi_heads.box_predictor.cls_score.in_features
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(in_features, num_classes, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            proposals, proposal_losses = self.fasterrcnn.rpn(images, features, targets)
            detections, detector_losses = self.fasterrcnn.roi_heads(features, proposals, images.image_sizes, targets)
            loss = proposal_losses + detector_losses
            return loss
        else:
            image_list = ImageList(images, [(images.shape[-2], images.shape[-1])])
            features = self.fasterrcnn.backbone([images.tensors])
            proposals, proposal_losses = self.fasterrcnn.rpn(images, features, targets)
            detections, detector_losses = self.fasterrcnn.roi_heads(features, proposals, images.image_sizes, targets)
            
            mask_features = [features[f] for f in self.fasterrcnn.roi_heads.in_features]
            mask_logits = self.mask_predictor(mask_features)
            
            result = {
                "detections": detections,
                "mask_logits": mask_logits,
            }
            return result


model = FasterRCNNWithMask(num_classes=3) 
model.eval()


image_path = "data/images/dog.jpg"
image = Image.open(image_path)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(image).unsqueeze(0)
# image_sizes = [(image.shape[-2], image.shape[-1])]
# images_listed = ImageList(input_tensor, image_sizes)
output = model(input_tensor)

boxes = output['detections'][0]['boxes']
labels = output['detections'][0]['labels']
mask_logits = output['mask_logits'][0]


masks = mask_logits > 0.5  

masked_image = input_tensor.copy()
draw = ImageDraw.Draw(masked_image)

for i in range(masks.shape[2]):
    mask = masks[:, :, i]
    mask = mask.squeeze().numpy()
    contours = find_contours(mask, 0.5)
    
    for contour in contours:
        contour = np.flip(contour, axis=1)
        draw.line(tuple(map(tuple, contour)), fill=(255, 0, 0), width=2)

# Display or save the result
masked_image.show()


visualized_masks = draw_mask(masks.float(), boxes.unsqueeze(0), labels.unsqueeze(0))

image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, label in zip(boxes, labels):
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

ax.imshow(visualized_masks.squeeze(0).permute(1, 2, 0), alpha=0.5, cmap='viridis')
plt.show()