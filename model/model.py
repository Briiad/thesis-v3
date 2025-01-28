import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CustomSSD(SSD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        """Custom loss combining Focal Loss and GIoU Loss"""
        # Classification loss (Focal Loss)
        cls_logits = head_outputs["cls_logits"]
        cls_targets = self._get_targets_from_matched_idxs(matched_idxs, targets)
        cls_loss = FocalLoss()(cls_logits, cls_targets)

        # Regression loss (GIoU Loss)
        bbox_regression = head_outputs["bbox_regression"]
        box_loss = torch.tensor(0.0, device=bbox_regression.device)
        for targets_per_image, bbox_regression_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, matched_idxs
        ):
            if matched_idxs_per_image.numel() == 0:
                continue

            matched_gt_boxes = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
            box_loss += generalized_box_iou_loss(
                bbox_regression_per_image[matched_idxs_per_image >= 0],
                matched_gt_boxes
            ).mean()

        return {
            "classification": cls_loss * 0.8,
            "bbox_regression": box_loss * 1.2 / len(targets)
        }

    def _get_targets_from_matched_idxs(self, matched_idxs, targets):
        # Reimplementation of the parent method to create classification targets
        labels = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            gt_classes = targets_per_image["labels"].to(dtype=torch.int64)
            # Assign labels: 0 for background (matched_idxs < 0), else corresponding class
            labels_per_image = gt_classes[matched_idxs_per_image.clip(min=0)]
            labels_per_image[matched_idxs_per_image < 0] = 0  # Background class
            labels.append(labels_per_image)
        return torch.cat(labels, dim=0)

class CustomBackboneWithFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        self.layer1 = nn.Sequential(*backbone[0:2])    # 16
        self.layer2 = nn.Sequential(*backbone[2:4])    # 24
        self.layer3 = nn.Sequential(*backbone[4:7])    # 32
        self.layer4 = nn.Sequential(*backbone[7:14])   # 96
        self.layer5 = nn.Sequential(*backbone[14:])    # 1280

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[16, 24, 32, 96, 1280],
            out_channels=256  # Reduced channels for efficiency
        )

    def forward(self, x):
        enc0 = self.layer1(x)
        enc1 = self.layer2(enc0)
        enc2 = self.layer3(enc1)
        enc3 = self.layer4(enc2)
        enc4 = self.layer5(enc3)
        return self.fpn({
            '0': enc0, '1': enc1, '2': enc2, 
            '3': enc3, '4': enc4
        })

def create_ssd_model(num_classes, pretrained_backbone=True):
    backbone = CustomBackboneWithFPN(pretrained=pretrained_backbone)
    
    anchor_generator = AnchorGenerator(
        sizes=((4, 8, 16), (8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    head = SSDHead(
        in_channels=[256] * 5,
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )

    return CustomSSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(640, 640),
        head=head,
        score_thresh=0.01
    )

if __name__ == '__main__':
    model = create_ssd_model(num_classes=7)
    model.eval()
    image = torch.randn(1, 3, 640, 640)
    output = model(image)
    print("Output shapes:", {k: v.shape for k, v in output[0].items()})
  