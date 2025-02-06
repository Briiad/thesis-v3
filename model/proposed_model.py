import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import generalized_box_iou, batched_nms, box_iou

class MobileNetV3BiFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT
          ).features
        self.layer1 = nn.Sequential(backbone[0], backbone[1])   # 16
        self.layer2 = backbone[2:4]                             # 24
        self.layer3 = backbone[4:7]                             # 40
        self.layer4 = backbone[7:13]                            # 112
        self.layer5 = backbone[13:]                             # 960
        
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
        
        # Top-down path
        self.td_conv1 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.td_conv2 = nn.Conv2d(in_channels[2], out_channels, 1)
        
        # Bottom-up path
        self.bu_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, features):
        # Unpack features
        p3, p4, p5, p6 = features['0'], features['1'], features['2'], features['3']
        
        # Top-down fusion
        p6_td = self.td_conv1(p6)
        p5_td = self.td_conv2(p5) + F.interpolate(p6_td, scale_factor=2)
        p4_td = p4 + F.interpolate(p5_td, scale_factor=2)
        p3_td = p3 + F.interpolate(p4_td, scale_factor=2)
        
        # Bottom-up fusion
        p4_bu = self.bu_conv(p4_td + F.max_pool2d(p3_td, kernel_size=2))
        p5_bu = self.bu_conv(p5_td + F.max_pool2d(p4_bu, kernel_size=2))
        p6_bu = self.bu_conv(p6_td + F.max_pool2d(p5_bu, kernel_size=2))
        
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
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 256),
            DepthwiseSeparableConv(256, 256),
            nn.Conv2d(256, num_anchors * num_classes, 3, padding=1)
        )
        
        # FCOS-style regression head
        self.reg_head = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 256),
            DepthwiseSeparableConv(256, 256),
            nn.Conv2d(256, num_anchors * 4, 3, padding=1)
        )
        
        # FCOS center-ness branch
        self.centerness = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 128),
            nn.Conv2d(128, num_anchors * 1, 3, padding=1)
        )

    def forward(self, x):
        cls_logits = []
        bbox_reg = []
        centerness = []
        
        for feature in x.values():
            cls_logits.append(self.cls_head(feature))
            reg_output = self.reg_head(feature)
            bbox_reg.append(reg_output)
            centerness.append(self.centerness(feature))
            
        return cls_logits, bbox_reg, centerness

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 
                           padding=1, groups=in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.pw(self.dw(x)))

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
        # Convert list of tensors to batched tensor
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
            
        # Feature extraction
        features = self.backbone(images)  # Remove .tensors access
        
        # Head outputs
        cls_logits, bbox_reg, centerness = self.head(features)
        
        # Post-processing
        anchors = self.anchor_generator(images, features)
        
        if self.training:
            return self.compute_loss(
                anchors, cls_logits, bbox_reg, centerness, targets)
        else:
            return self.post_process(
                anchors, cls_logits, bbox_reg, centerness)

    def compute_loss(self, anchors, cls_logits, bbox_reg, centerness, targets):
        # Initialize loss dictionaries
        losses = {
            'classification': torch.tensor(0., device=cls_logits[0].device),
            'regression': torch.tensor(0., device=cls_logits[0].device),
            'centerness': torch.tensor(0., device=cls_logits[0].device)
        }
        
        # Process each feature level
        for level_idx, (anchors_level, cls_logits_level, 
                        bbox_reg_level, centerness_level) in enumerate(zip(
                            anchors, cls_logits, bbox_reg, centerness)):
            
            # Match anchors to targets
            matched_gt_boxes, matched_gt_labels, positive_mask = self.match_anchors(
                anchors_level, targets, level_idx)
            
            # Skip levels with no positive samples
            num_positives = positive_mask.sum()
            if num_positives == 0:
                continue

            # Classification loss (Focal Loss)
            cls_logits_level = cls_logits_level.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            cls_targets = torch.full_like(cls_logits_level[:, 0], -1, dtype=torch.long)
            cls_targets[positive_mask] = matched_gt_labels
            losses['classification'] += self.focal_loss(
                cls_logits_level, cls_targets, 
                alpha=self.focal_loss_alpha, 
                gamma=self.focal_loss_gamma
            )

            # Regression loss (GIoU)
            pred_boxes = self.decode_boxes(anchors_level[positive_mask], 
                                          bbox_reg_level.permute(0, 2, 3, 1).reshape(-1, 4)[positive_mask])
            losses['regression'] += 1 - torch.diag(generalized_box_iou(
                pred_boxes, 
                matched_gt_boxes[positive_mask]
            )).mean()

            # Centerness loss (BCEWithLogitsLoss)
            centerness_level = centerness_level.permute(0, 2, 3, 1).reshape(-1)
            centerness_targets = self.compute_centerness_targets(
                anchors_level[positive_mask], 
                matched_gt_boxes[positive_mask]
            )
            losses['centerness'] += F.binary_cross_entropy_with_logits(
                centerness_level[positive_mask], 
                centerness_targets, 
                reduction='mean'
            )

        # Apply loss weights
        total_loss = (
            1.0 * losses['classification'] +
            1.0 * losses['regression'] +
            1.0 * losses['centerness']
        )
        
        return {'total_loss': total_loss, **losses}

    def post_process(self, anchors, cls_logits, bbox_reg, centerness):
        detections = []
        batch_size = cls_logits[0].shape[0]
        
        for batch_idx in range(batch_size):
            batch_boxes = []
            batch_scores = []
            batch_labels = []
            
            # Process each feature level
            for level_idx, (anchors_level, cls_logits_level, 
                            bbox_reg_level, centerness_level) in enumerate(zip(
                                anchors, cls_logits, bbox_reg, centerness)):
                
                # Decode boxes
                level_boxes = self.decode_boxes(
                    anchors_level, 
                    bbox_reg_level[batch_idx].permute(1, 2, 0).reshape(-1, 4)
                )
                
                # Apply centerness
                level_centerness = torch.sigmoid(
                    centerness_level[batch_idx].permute(1, 2, 0).reshape(-1)
                )
                level_scores = torch.sigmoid(
                    cls_logits_level[batch_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
                ) * level_centerness[:, None]
                
                # Filter by score threshold
                keep = level_scores.max(dim=1).values > 0.01
                level_boxes = level_boxes[keep]
                level_scores = level_scores[keep]
                
                batch_boxes.append(level_boxes)
                batch_scores.append(level_scores)
            
            # Combine all levels
            boxes = torch.cat(batch_boxes)
            scores = torch.cat(batch_scores)
            labels = scores.argmax(dim=1)
            scores = scores.max(dim=1).values
            
            # Apply NMS
            keep = batched_nms(
                boxes, scores, labels, 
                iou_threshold=0.25
            )
            keep = keep[:200]
            
            detections.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep]
            })
        
        return detections

    # Helper methods
    def match_anchors(self, anchors, targets, level_idx):
        iou_matrix = box_iou(anchors, targets['boxes'])
        max_iou, matched_idxs = iou_matrix.max(dim=1)
        
        # Create positive mask (IoU > 0.5)
        positive_mask = max_iou > 0.5
        matched_boxes = targets['boxes'][matched_idxs[positive_mask]]
        matched_labels = targets['labels'][matched_idxs[positive_mask]]
        
        return matched_boxes, matched_labels, positive_mask

    def decode_boxes(self, anchors, deltas):
        # Convert anchor+delta to boxes
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0] * widths
        dy = deltas[:, 1] * heights
        dw = deltas[:, 2] * widths
        dh = deltas[:, 3] * heights
        
        pred_ctr_x = ctr_x + dx
        pred_ctr_y = ctr_y + dy
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        return torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], dim=1)

    def compute_centerness_targets(self, anchors, gt_boxes):
        # FCOS-style centerness calculation
        left_right = gt_boxes[:, [0, 2]] - anchors[:, [0, 2]]
        top_bottom = gt_boxes[:, [1, 3]] - anchors[:, [1, 3]]
        centerness = torch.sqrt(
            (left_right.min(dim=1).values / left_right.max(dim=1).values).clamp(min=0) *
            (top_bottom.min(dim=1).values / top_bottom.max(dim=1).values).clamp(min=0)
        )
        return centerness

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** gamma * ce_loss
        
        # Alpha weighting
        alpha_factor = torch.ones_like(targets) * alpha
        alpha_factor = torch.where(targets == 1, alpha_factor, 1 - alpha_factor)
        return (alpha_factor * loss).mean()

if __name__ == '__main__':
    model = create_proposed_model(num_classes=7)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")