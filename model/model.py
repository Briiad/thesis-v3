import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import boxes as box_ops

# ======== Key Modification 1: BiFPN Implementation ========
class BiFPN(nn.Module):
    """Lightweight Bidirectional Feature Pyramid Network"""
    def __init__(self, feature_channels=[16, 24, 32, 96, 1280], out_channels=128):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(2))  # Weighted fusion parameters
        self.w2 = nn.Parameter(torch.ones(3))
        
        # Top-down pathway
        self.td_conv1 = nn.Sequential(
            nn.Conv2d(feature_channels[-1], out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)  # Depthwise
        )
        self.td_conv2 = nn.Conv2d(feature_channels[-2], out_channels, 1)
        self.td_conv3 = nn.Conv2d(feature_channels[-3], out_channels, 1)
        
        # Bottom-up pathway
        self.bu_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bu_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        # Unpack features from MobileNetV2
        c2, c3, c4, c5, c6 = features.values()
        
        # Top-down fusion
        p6 = self.td_conv1(c6)
        p5 = self.td_conv2(c5) + F.interpolate(p6, scale_factor=2)
        p4 = self.td_conv3(c4) + F.interpolate(p5, scale_factor=2)
        
        # Bottom-up fusion
        n4 = self.bu_conv1(p4)
        n5 = self.bu_conv2(p5 + F.max_pool2d(n4, kernel_size=2))
        n6 = self.bu_conv2(p6 + F.max_pool2d(n5, kernel_size=2))
        
        return {'0': n4, '1': n5, '2': n6}  # 3 output levels

# ======== Key Modification 2: Updated Backbone ========
class CustomBackboneWithBiFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        self.layer1 = nn.Sequential(*backbone[0:2])  # 16
        self.layer2 = nn.Sequential(*backbone[2:4])  # 24
        self.layer3 = nn.Sequential(*backbone[4:7])  # 32
        self.layer4 = nn.Sequential(*backbone[7:14])  # 96
        self.layer5 = nn.Sequential(*backbone[14:])  # 1280

        self.bifpn = BiFPN(feature_channels=[16, 24, 32, 96, 1280], out_channels=256)

    def forward(self, x):
        enc0 = self.layer1(x)
        enc1 = self.layer2(enc0)
        enc2 = self.layer3(enc1)
        enc3 = self.layer4(enc2)
        enc4 = self.layer5(enc3)
        return self.bifpn({
            '0': enc0, '1': enc1, '2': enc2, 
            '3': enc3, '4': enc4
        })

def create_ssd_model(num_classes, pretrained_backbone=True):
    # Use new BiFPN backbone
    backbone = CustomBackboneWithBiFPN(pretrained=pretrained_backbone)
    
    # ======== Key Modification 4: Optimized Anchors ========
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32), (16, 32, 64), (32, 64, 128)),  # For 3 BiFPN levels
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 3  # Match number of BiFPN levels
    )
    
    # ======== Key Modification 5: Adjusted SSD Head ========
    head = SSDHead(
        in_channels=[128] * 3,  # Matches BiFPN output channels
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )

    return SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        head=head,
        iou_thresh=0.15,  # Relaxed threshold
        num_classes=num_classes,
        size=(640, 640),
        nms_thresh=0.25,
        score_thresh=0.01,  # Lower threshold for better recall
        detections_per_img=200
    )

if __name__ == '__main__':
    model = create_ssd_model(num_classes=7)
    model.eval()
    image = torch.randn(1, 3, 640, 640)
    output = model(image)
    print("Output shapes:", {k: v.shape for k, v in output[0].items()})