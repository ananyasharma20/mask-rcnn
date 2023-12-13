import torch
import torchvision
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from torch.nn import functional as F
from torchvision import ops
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from utils import * 
from model import *
from torchvision.models.detection.image_list import ImageList
from torch.utils.data import DataLoader, Dataset
from ObjectDetection import ObjectDetectionDataset

# We referenced the following code while writing this file: 
# https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0
# https://github.com/wingedrasengan927/pytorch-tutorials/tree/master/Object%20Detection

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes
    if mode == 'a2p':
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes

def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=3):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor='yellow', alpha=0.5))
        c += 1
        
    return fig, ax

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0) \
                              , anc_pts_y.size(dim=0), n_anc_boxes, 4) # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]
    
    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale
                    
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
    print("ANCHOR BASE DONE")
    return anc_base


def display_grid(x_points, y_points, fig, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')
            
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
        
    return fig, ax

def gen_anc_centers(out_size):
    out_h, out_w = out_size
    
    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5
    
    return anc_pts_x, anc_pts_y


def display_img(img_data, fig, axes):
    print("IMAGE DATA SHAPE",len(img_data[0]))
    for i, img in enumerate(img_data):
        print("IMAGE IN DISPLAY IMG", img.shape)
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
            print("NEW VERSION OF IMG", img)
        print("DIPSPLAING THE IMAGE")
        axes[i].imshow(img)
    return fig, axes

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
        
    def forward(self, img_data):
        return self.backbone(img_data)
    
def decode_boxes(rpn_output, anchors):
    print("RPN OUTPUT", rpn_output)
    """
    Decode RPN offsets to obtain final bounding box coordinates.
    
    Args:
    - rpn_output: Tensor of RPN output (offsets).
    - anchors: Tensor of anchor boxes.

    Returns:
    - decoded_boxes: Tensor of decoded bounding boxes.
    """

    widths = anchors[0][:, 2] - anchors[0][:, 0]
    heights = anchors[0][:, 3] - anchors[0][:, 1]
    ctr_x = anchors[0][:, 0] + 0.5 * widths
    ctr_y = anchors[0][:, 1] + 0.5 * heights

    dx = rpn_output[:, 0]
    dy = rpn_output[:, 1]
    dw = rpn_output[:, 2]
    dh = rpn_output[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # Calculate final bounding box coordinates
    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return decoded_boxes

class BackboneWithoutGAP(nn.Module):
    def __init__(self):
        super(BackboneWithoutGAP, self).__init__()
        # Load pre-trained ResNet-50 model
        resnet = resnet50(pretrained=True)
        # Remove the global average pooling (GAP) layer
        self.features = nn.Sequential(
            *list(resnet.children())[:-2],  # Remove the GAP and fully connected layers
            nn.AdaptiveAvgPool2d(1)  # Add an adaptive pooling layer to replace the removed GAP
        )

    def forward(self, x):
        return self.features(x)
    
def training_loop(model, learning_rate, train_dataloader, n_epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
            
            # forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        loss_list.append(total_loss)
        
    return loss_list

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()

        self.backbone = FeatureExtractor()
        backbone_out_channels = 2048
        out_h, out_w = 800, 800
        print("Number of output channels:", backbone_out_channels)
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(out_h, out_w))
        anc_scales = [2, 4, 6]
        anc_ratios = [0.5, 1, 1.5]
        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, (out_h, out_w))
        anc_boxes_all = anc_base.repeat(images.size(dim=0), 1, 1, 1, 1)

        self.anchor_generator = AnchorGenerator(sizes=[(32, 64, 128)],aspect_ratios=[0.5, 1.0, 2.0])

        # ROI Pooling
        self.roi_pooling = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        # Heads for classification, bounding box regression, and mask prediction
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(backbone_out_channels, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, images, targets=None):
        # Backbone
        features = self.backbone(images)
        out_c, out_h, out_w = features.size(dim=1), features.size(dim=2), features.size(dim=3)
        width_scale_factor = images[0].shape[-2] // out_w
        height_scale_factor = images[0].shape[-1] // out_h
    

        filters_data =[filters[0].detach().numpy() for filters in features[:2]]
        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4))
        filters_data =[filters[0].detach().numpy() for filters in features[:2]]
        features_plot = display_img(filters_data, fig, axes)
        plt.show()
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(out_h, out_w))

        anc_pts_x_proj = anc_pts_x.clone() * width_scale_factor 
        anc_pts_y_proj = anc_pts_y.clone() * height_scale_factor
        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
        fig, axes = display_img(images, fig, axes)
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])
        # plt.show()


        anc_scales = [2, 4, 6]
        anc_ratios = [0.5, 1, 1.5]
        n_anc_boxes = len(anc_scales) * len(anc_ratios) # number of anchor boxes for each anchor point

        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, (out_h, out_w))
        anc_boxes_all = anc_base.repeat(images.size(dim=0), 1, 1, 1, 1)
        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

        fig, axes = display_img(images, fig, axes)

        # project anchor boxes to the image
        anc_pts_x_proj = anc_pts_x.clone() * width_scale_factor 
        anc_pts_y_proj = anc_pts_y.clone() * height_scale_factor
        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
        fig, axes = display_img(images, fig, axes)
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])
        plt.show()

        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
        fig, axes = display_img(images, fig, axes)
        anc_boxes_proj = project_bboxes(anc_boxes_all, width_scale_factor, height_scale_factor, mode='a2p')
        sp_1 = [5, 8]
        sp_2 = [12, 15]
        bboxes_1 = anc_boxes_proj[0][sp_1[0], sp_1[1]]
        bboxes_2 = anc_boxes_proj[0][sp_2[0], sp_2[1]]
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0], (anc_pts_x_proj[sp_1[0]], anc_pts_y_proj[sp_1[1]]))
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1], (anc_pts_x_proj[sp_2[0]], anc_pts_y_proj[sp_2[1]]))
        fig, _ = display_bbox(bboxes_1, fig, axes[0])
        fig, _ = display_bbox(bboxes_2, fig, axes[0])
        plt.show()

        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

        fig, axes = display_img(images, fig, axes)

        # plot feature grid
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
        fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])

        # plot all anchor boxes
        print("PLOTTING ALL ANCHOR BOXES")
        for x in range(anc_pts_x_proj.size(dim=0)):
            for y in range(anc_pts_y_proj.size(dim=0)):
                bboxes = anc_boxes_proj[0][x, y]
                fig, _ = display_bbox(bboxes, fig, axes[0], line_width=1)
                fig, _ = display_bbox(bboxes, fig, axes[1], line_width=1)
        # plt.show()
     
        #plot positive and negative anchor boxes
        pos_thresh = 0.7
        neg_thresh = 0.3
        # project gt bboxes onto the feature map
        gt_bboxes_proj = project_bboxes(gt_bboxes_all, width_scale_factor, height_scale_factor, mode='p2a')      


        positive_anc_ind, negative_anc_ind, GT_conf_scores, \
        GT_offsets, GT_class_pos, positive_anc_coords, \
        negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes_all, pos_thresh, neg_thresh)

        # project anchor coords to the image space
        pos_anc_proj = project_bboxes(positive_anc_coords, width_scale_factor, height_scale_factor, mode='a2p')
        neg_anc_proj = project_bboxes(negative_anc_coords, width_scale_factor, height_scale_factor, mode='a2p')

        # grab +ve and -ve anchors for each image separately
        anc_idx_1 = torch.where(positive_anc_ind_sep == 0)[0]
        anc_idx_2 = torch.where(positive_anc_ind_sep == 1)[0]

        pos_anc_1 = pos_anc_proj[anc_idx_1]
        pos_anc_2 = pos_anc_proj[anc_idx_2]

        neg_anc_1 = neg_anc_proj[anc_idx_1]
        neg_anc_2 = neg_anc_proj[anc_idx_2]

        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

        fig, axes = display_img(images, fig, axes)

        # plot groundtruth bboxes
        fig, _ = display_bbox(gt_bboxes_all[0], fig, axes[0])

        # plot positive anchor boxes
        fig, _ = display_bbox(pos_anc_1, fig, axes[0], color='g')

        # plot negative anchor boxes
        fig, _ = display_bbox(neg_anc_1, fig, axes[0], color='r')
        plt.show()

        rpn_output = self.rpn(features)
        proposals = det_utils.proposals_from_rpn_output(
            rpn_output, boxes, img_sizes, self.rpn.anchor_generator
        )
        print(features, proposals, image.shape, "ARGS TO POOLING")

        box_features = self.roi_pooling(features, proposals, image.shape)
        box_features = box_features.view(box_features.size(0), -1)

        class_scores = self.classification_head(box_features)

        box_regression = self.bbox_head(box_features)

        mask_features = self.mask_head(features[-1])
        mask_scores = F.interpolate(mask_features, size=proposals.shape[-2:], mode='bilinear', align_corners=False)

        if self.training:
            # Training mode
            losses = {}

            return losses
        else:
            # Inference mode
            result = {
                'boxes': box_regression,
                'labels': class_scores.argmax(dim=1),
                'masks': mask_scores
            }
            return result

# Instantiate the model
num_classes = 21  
model = MaskRCNN(num_classes)


model.eval()

# Process images
image1_path = "data/images/dog.jpg"
image2_path = "data/images/bike.jpg"
image1 = Image.open(image1_path).convert("RGB")
img_size = (image1.size[0], image1.size[1])
image2 = Image.open(image2_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_dir = os.path.join("data", "images")
img_width, img_height = 800, 800
name2idx = {'pad': -1, 'dog': 0, 'bike': 1, 'person': 2, 'lamp': 3, 'tree':4}
idx2name = {v:k for k, v in name2idx.items()}
dataset = ObjectDetectionDataset("annotationsDog.xml", image_dir, (img_height, img_width), name2idx)
od_dataloader = DataLoader(dataset, batch_size=1)
for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
    img_data_all = img_batch
    gt_bboxes_all = gt_bboxes_batch
    gt_classes_all = gt_classes_batch
    break
    
img_data_all = img_data_all[:1]
gt_bboxes_all = gt_bboxes_all[:1]
gt_classes_all = gt_classes_all[:1]
# get class names
gt_class_1 = gt_classes_all[0].long()
gt_class_1 = [idx2name[idx.item()] for idx in gt_class_1]

nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

fig, axes = display_img(img_data_all, fig, axes)
fig, _ = display_bbox(gt_bboxes_all[0], fig, axes[0], classes=gt_class_1)
plt.show()


# positive_anc_ind, negative_anc_ind, GT_conf_scores, \
# GT_offsets, GT_class_pos, positive_anc_coords, \
# negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes_all, pos_thresh, neg_thresh)


input_image1 = transform(image1)
input_image1 = input_image1.unsqueeze(0)
input_image2 = transform(image2)
input_image2 = input_image2.unsqueeze(0)
images = [input_image1, input_image2]

out_size = (800,800)

learning_rate = 1e-3
n_epochs = 100

with torch.no_grad():
    predictions = model(input_image1)

boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
masks = predictions[0]['masks'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(np.array(input_image1))

for box, label, score, mask in zip(boxes, labels, scores, masks):
    # Draw bounding box
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Display class label and score
    ax.text(box[0], box[1], f'Class: {label}, Score: {score:.2f}', color='r')

    mask = mask[0]  
    mask = np.where(mask > 0.5, 1, 0)  
    mask_color = np.random.rand(3)  
    ax.imshow(mask, alpha=0.3, cmap='viridis', facecolor=mask_color)

plt.axis('off')
plt.show()
