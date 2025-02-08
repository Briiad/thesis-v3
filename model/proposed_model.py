import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.fcos import FCOSHead
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import batched_nms, generalized_box_iou_loss
from torchvision.models.detection.image_list import ImageList

##############################
# Backbone: FPN with BiFPN
##############################
class MobileNetV3BiFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone using MobileNetV3 (pre-trained weights)
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        self.layer1 = nn.Sequential(backbone[0], backbone[1])  # Example: 16 channels
        self.layer2 = backbone[2:4]   # 24 channels
        self.layer3 = backbone[4:7]   # 40 channels
        self.layer4 = backbone[7:13]  # 112 channels
        self.layer5 = backbone[13:]   # 960 channels

        # Build a BiFPN on top of the backbone layers
        self.bifpn = nn.ModuleList([
            BiFPNBlock(in_channels=[24, 40, 112, 960], out_channels=128)
        ])

    def forward(self, x):
        # Feature extraction from backbone
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)

        # Prepare feature dictionary for BiFPN
        features = {'0': c3, '1': c4, '2': c5, '3': c6}
        for bifpn in self.bifpn:
            features = bifpn(features)
        return features

class BiFPNBlock(nn.Module):
    """Bi-directional Feature Pyramid Network Block"""
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        # Top-down pathway convolutions
        self.td_conv1 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.td_conv2 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.td_conv3 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.td_conv4 = nn.Conv2d(in_channels[0], out_channels, 1)
        # Bottom-up pathway convolutions
        self.bu_conv1 = nn.Conv2d(out_channels, out_channels, 1)
        self.bu_conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bu_conv3 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, features):
        p3, p4, p5, p6 = features['0'], features['1'], features['2'], features['3']
        # Top-down fusion
        p6_td = self.td_conv1(p6)
        p5_td = self.td_conv2(p5) + F.interpolate(p6_td, scale_factor=2)
        p4_td = self.td_conv3(p4) + F.interpolate(p5_td, scale_factor=2)
        p3_td = self.td_conv4(p3) + F.interpolate(p4_td, scale_factor=2)
        # Bottom-up fusion
        p4_bu = self.bu_conv1(p4_td + F.max_pool2d(p3_td, kernel_size=2))
        p5_bu = self.bu_conv2(p5_td + F.max_pool2d(p4_bu, kernel_size=2))
        p6_bu = self.bu_conv3(p6_td + F.max_pool2d(p5_bu, kernel_size=2))
        return {
            '0': self.conv(p3_td),
            '1': self.conv(p4_bu),
            '2': self.conv(p5_bu),
            '3': self.conv(p6_bu)
        }

#########################################
# New Detection Head: Anchor-Free FCOS
#########################################
class FCOSHybridHead(nn.Module):
    """
    A head that uses FCOS's anchor-free mechanism to produce both
    classification and regression outputs. This avoids having to
    align anchor counts with predictions.
    """
    def __init__(self, in_channels=128, num_classes=7):
        super().__init__()
        # FCOSHead from torchvision is used with num_anchors=1 (anchor-free)
        self.fcos_head = FCOSHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_anchors=1
        )

    def forward(self, features):
        # Convert feature dict to a list (ordered by scale)
        features_list = list(features.values())
        outputs = self.fcos_head(features_list)
        # Expected keys: "cls_logits", "bbox_regression", "bbox_ctrness"
        return outputs["cls_logits"], outputs["bbox_regression"], outputs["bbox_ctrness"]

#########################################
# Hybrid Detection Model (Anchor-Free)
#########################################
class HybridDetectionModel(nn.Module):
    def __init__(self, backbone, head, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes
        # Loss parameters (adjust as needed for FCOS-style targets)
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.center_loss_weight = 0.1

    def forward(self, images, targets=None):
        # Expecting images as a tensor (or list converted to tensor)
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        features = self.backbone(images)
        cls_logits, bbox_reg, centerness = self.head(features)
        
        if self.training:
            return self.compute_loss(cls_logits, bbox_reg, centerness, targets)
        else:
            return self.post_process(cls_logits, bbox_reg, centerness)

    def compute_loss(self, cls_logits, bbox_reg, centerness, targets):
        # Note: In a full FCOS implementation, target assignment is performed per feature map location.
        # The following is a simplified placeholder implementation.
        B, N, num_classes = cls_logits.shape
        cls_logits = cls_logits.reshape(-1, num_classes)
        bbox_reg = bbox_reg.reshape(-1, 4)
        centerness = centerness.reshape(-1)
        
        # Assume targets are provided as lists of dicts with keys 'labels' and 'boxes'
        gt_classes = torch.cat([t['labels'] for t in targets], dim=0)
        gt_bboxes = torch.cat([t['boxes'] for t in targets], dim=0)
        
        ce_loss = F.cross_entropy(cls_logits, gt_classes, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * ce_loss).mean()
        iou_loss = generalized_box_iou_loss(bbox_reg, gt_bboxes, reduction='mean')
        center_loss = F.binary_cross_entropy_with_logits(centerness, torch.ones_like(centerness))
        
        total_loss = focal_loss + iou_loss + self.center_loss_weight * center_loss
        return {"loss": total_loss, "focal_loss": focal_loss, "iou_loss": iou_loss, "center_loss": center_loss}

    def post_process(self, cls_logits, bbox_reg, centerness, conf_threshold=0.05, iou_threshold=0.5):
        # Convert logits to probabilities
        cls_probs = torch.sigmoid(cls_logits)  # Shape: [B*N, num_classes]
        centerness_probs = torch.sigmoid(centerness)  # Shape: [B*N]
        
        # Multiply to obtain final scores
        scores = cls_probs * centerness_probs.unsqueeze(-1)
        max_scores, labels = scores.max(dim=-1)
        keep = max_scores > conf_threshold

        scores = scores[keep]
        bbox_reg = bbox_reg[keep]
        max_scores = max_scores[keep]
        labels = labels[keep]

        # Apply non-maximum suppression (NMS)
        keep_indices = batched_nms(bbox_reg, max_scores, labels, iou_threshold)
        return [{
            "boxes": bbox_reg[keep_indices],
            "scores": scores[keep_indices],
            "labels": labels[keep_indices]
        }]

#########################################
# Model Factory Function
#########################################
def create_proposed_model(num_classes):
    # Anchor generator is no longer needed in the anchor-free setup.
    backbone = MobileNetV3BiFPN()
    head = FCOSHybridHead(num_classes=num_classes)
    return HybridDetectionModel(
        backbone=backbone,
        head=head,
        num_classes=num_classes
    )

#########################################
# Example Usage
#########################################
if __name__ == '__main__':
    model = create_proposed_model(num_classes=7)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Eval Test
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512)  # Example input
    output = model(dummy_input)
    print("Forward pass successful!")
