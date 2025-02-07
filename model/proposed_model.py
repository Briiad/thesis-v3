import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.fcos import FCOSHead
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import batched_nms, generalized_box_iou_loss
from torchvision.models.detection.image_list import ImageList


class MobileNetV3BiFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        self.layer1 = nn.Sequential(backbone[0], backbone[1])  # 16
        self.layer2 = backbone[2:4]  # 24
        self.layer3 = backbone[4:7]  # 40
        self.layer4 = backbone[7:13]  # 112
        self.layer5 = backbone[13:]  # 960

        # BiFPN Configuration
        self.bifpn = nn.ModuleList([
            BiFPNBlock(in_channels=[24, 40, 112, 960], out_channels=128)
        ])

    def forward(self, x):
        # Feature Extraction
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)

        # BiFPN Processing
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
        # Top-down path: Add 1x1 convolutions to match channel dimensions
        self.td_conv1 = nn.Conv2d(in_channels[3], out_channels, 1)  # For p6
        self.td_conv2 = nn.Conv2d(in_channels[2], out_channels, 1)  # For p5
        self.td_conv3 = nn.Conv2d(in_channels[1], out_channels, 1)  # For p4
        self.td_conv4 = nn.Conv2d(in_channels[0], out_channels, 1)  # For p3

        # Bottom-up path: Add 1x1 convolutions to match channel dimensions
        self.bu_conv1 = nn.Conv2d(out_channels, out_channels, 1)  # For p4
        self.bu_conv2 = nn.Conv2d(out_channels, out_channels, 1)  # For p5
        self.bu_conv3 = nn.Conv2d(out_channels, out_channels, 1)  # For p6

    def forward(self, features):
        # Unpack features
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

class HybridHead(nn.Module):
    """Combines RetinaNet classification with FCOS regression"""
    def __init__(self, in_channels=128, num_classes=7, num_anchors=9):
        super().__init__()
        # Shared parameters
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # RetinaNet-style classification head
        self.cls_head = RetinaNetHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

        # FCOS-style regression head
        self.reg_head = FCOSHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_anchors=1
        )

    def forward(self, x):
        # Convert features dict to ordered list
        features_list = list(x.values())
        
        # Process through heads
        cls_logits = self.cls_head(features_list)["cls_logits"]
        reg_outputs = self.reg_head(features_list)
        
        return cls_logits, reg_outputs["bbox_regression"], reg_outputs["bbox_ctrness"]


def create_proposed_model(num_classes):
    # Anchor configuration for small objects
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 4
    )

    # Model components
    backbone = MobileNetV3BiFPN()
    head = HybridHead(num_classes=num_classes)

    return HybridDetectionModel(
        backbone=backbone,
        head=head,
        anchor_generator=anchor_generator,
        num_classes=num_classes
    )


class HybridDetectionModel(nn.Module):
    def __init__(self, backbone, head, anchor_generator, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes

        # Loss parameters
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.center_loss_weight = 0.1

    def forward(self, images, targets=None):
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        features = self.backbone(images)
        feature_maps = list(features.values())
        cls_logits, bbox_reg, centerness = self.head(features)
        
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        image_list = ImageList(images, image_sizes)
        anchors = self.anchor_generator(image_list, feature_maps)

        if self.training:
            return self.compute_loss(anchors, cls_logits, bbox_reg, centerness, targets)
        else:
            return self.post_process(anchors, cls_logits, bbox_reg, centerness)

    def compute_loss(self, anchors, cls_logits, bbox_reg, centerness, targets):
        # Flatten all anchors and predictions
        num_anchors = len(anchors)
        cls_logits = cls_logits.view(-1, self.num_classes)
        bbox_reg = bbox_reg.view(-1, 4)
        centerness = centerness.view(-1)

        # Extract ground truth values
        gt_classes = torch.cat([t['labels'] for t in targets], dim=0)
        gt_bboxes = torch.cat([t['boxes'] for t in targets], dim=0)
        
        # Focal Loss for classification
        ce_loss = F.cross_entropy(cls_logits, gt_classes, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * ce_loss).mean()
        
        # IoU Loss for bbox regression
        iou_loss = generalized_box_iou_loss(bbox_reg, gt_bboxes, reduction='mean')
        
        # Centerness loss
        center_loss = F.binary_cross_entropy_with_logits(centerness, torch.ones_like(centerness))
        
        # Total loss
        total_loss = focal_loss + iou_loss + self.center_loss_weight * center_loss
        return {"loss": total_loss, "focal_loss": focal_loss, "iou_loss": iou_loss, "center_loss": center_loss}

    def post_process(self, anchors, cls_logits, bbox_reg, centerness, conf_threshold=0.05, iou_threshold=0.5):
        cls_probs = torch.sigmoid(cls_logits)
        centerness_probs = torch.sigmoid(centerness)
        scores = cls_probs * centerness_probs.unsqueeze(-1)
        
        # Filter out low-confidence predictions
        keep = scores > conf_threshold
        scores, bbox_reg, anchors = scores[keep], bbox_reg[keep], anchors[keep]
        
        # Apply Non-Maximum Suppression (NMS)
        keep_indices = batched_nms(bbox_reg, scores.max(dim=1)[0], keep.nonzero()[:, 1], iou_threshold)
        
        # Format output
        output = [{
            "boxes": bbox_reg[keep_indices],
            "scores": scores[keep_indices],
            "labels": keep.nonzero()[:, 1][keep_indices]
        }]
        return output


if __name__ == '__main__':
    model = create_proposed_model(num_classes=7)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512)  # Example input (batch_size=1, channels=3, height=512, width=512)
    output = model(dummy_input)
    print("Forward pass successful!")