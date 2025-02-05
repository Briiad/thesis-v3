import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large

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
        self.layer5 = backbone[13:]                             # 1280
        
        # BiFPN Configuration
        self.bifpn = nn.ModuleList([
            BiFPNBlock(in_channels=[24, 40, 112, 1280], out_channels=128)
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
        # Feature extraction
        features = self.backbone(images)
        
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
        # Combined focal loss + GIoU loss + center-ness loss
        # Implementation details depend on your target format
        pass

    def post_process(self, anchors, cls_logits, bbox_reg, centerness):
        # Combine predictions with center-ness weighting
        # Apply NMS and thresholding
        pass

if __name__ == '__main__':
    model = create_proposed_model()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")