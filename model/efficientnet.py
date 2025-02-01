import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator

class EfficientBiFPN(nn.Module):
    """BiFPN optimized for EfficientNet features"""
    def __init__(self, feature_channels=[16, 24, 40, 112, 320], out_channels=160):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(3))
        self.w2 = nn.Parameter(torch.ones(3))
        
        # Top-down pathway
        self.td_conv1 = nn.Conv2d(feature_channels[-1], out_channels, 1)
        self.td_conv2 = nn.Conv2d(feature_channels[-2], out_channels, 1)
        self.td_conv3 = nn.Conv2d(feature_channels[-3], out_channels, 1)
        
        # Depthwise separable convs
        self.bu_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        self.bu_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, features):
        c3, c4, c5, c6, c7 = features.values()
        p7 = self.td_conv1(c7)
        p6 = self.td_conv2(c6) + F.interpolate(p7, scale_factor=2)
        p5 = self.td_conv3(c5) + F.interpolate(p6, scale_factor=2)
        
        n5 = self.bu_conv1(p5)
        n6 = self.bu_conv2(p6 + F.max_pool2d(n5, kernel_size=2))
        n7 = self.bu_conv2(p7 + F.max_pool2d(n6, kernel_size=2))
        
        return {'0': n5, '1': n6, '2': n7}

class EfficientNetB0Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.efficientnet_b0(pretrained=pretrained).features
        self.layer1 = backbone[1]   # 16
        self.layer2 = backbone[2]   # 24
        self.layer3 = backbone[3]   # 40
        self.layer4 = backbone[4]   # 112
        self.layer5 = backbone[5]   # 320
        
        self.bifpn = EfficientBiFPN(feature_channels=[16, 24, 40, 112, 320], out_channels=160)

    def forward(self, x):
        enc1 = self.layer1(x)
        enc2 = self.layer2(enc1)
        enc3 = self.layer3(enc2)
        enc4 = self.layer4(enc3)
        enc5 = self.layer5(enc4)
        return self.bifpn({'0': enc1, '1': enc2, '2': enc3, '3': enc4, '4': enc5})

def create_efficientnet_ssd(num_classes):
    anchor_generator = AnchorGenerator(
        sizes=((4, 8, 16), (8, 16, 32), (16, 32, 64)),
        aspect_ratios=((0.2, 0.5, 1.0, 2.0, 5.0),) * 3
    )
    
    head = SSDHead(
        in_channels=[160] * 3,
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )
    
    return SSD(
        backbone=EfficientNetB0Backbone(),
        anchor_generator=anchor_generator,
        head=head,
        num_classes=num_classes,
        size=(640, 640),
        iou_thresh=0.15,
        nms_thresh=0.25,
        score_thresh=0.01,
        detections_per_img=300
    )