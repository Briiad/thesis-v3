import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import boxes as box_ops

class BiFPN(nn.Module):
    """Modified BiFPN for MobileNetV3-Large"""
    def __init__(self, feature_channels=[16, 24, 40, 112, 1280], out_channels=128):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(3))
        
        # Top-down pathway
        self.td_conv1 = nn.Conv2d(feature_channels[-1], out_channels, 1)
        self.td_conv2 = nn.Conv2d(feature_channels[-2], out_channels, 1)
        self.td_conv3 = nn.Conv2d(feature_channels[-3], out_channels, 1)
        
        # Bottom-up with depthwise
        self.bu_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        self.bu_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, features):
        c2, c3, c4, c5, c6 = features.values()
        p6 = self.td_conv1(c6)
        p5 = self.td_conv2(c5) + F.interpolate(p6, scale_factor=2)
        p4 = self.td_conv3(c4) + F.interpolate(p5, scale_factor=2)
        
        n4 = self.bu_conv1(p4)
        n5 = self.bu_conv2(p5 + F.max_pool2d(n4, kernel_size=2))
        n6 = self.bu_conv2(p6 + F.max_pool2d(n5, kernel_size=2))
        
        return {'0': n4, '1': n5, '2': n6}

class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
        
        # Correct layer slicing for MobileNetV3-Large
        self.layer1 = nn.Sequential(backbone[0], backbone[1])   # out: 16
        self.layer2 = backbone[2:4]                             # out: 24
        self.layer3 = backbone[4:7]                             # out: 40
        self.layer4 = backbone[7:13]                            # out: 112
        self.layer5 = nn.Sequential(
            backbone[13],       # out: 160
            backbone[14],       # out: 160
            backbone[15],       # out: 160
            backbone[16]        # out: 960 (final features)
        )
        
        # Update BiFPN input channels to match actual outputs
        self.bifpn = BiFPN(
            feature_channels=[16, 24, 40, 112, 960],  # Changed last channel to 960
            out_channels=128
        )

    def forward(self, x):
        enc0 = self.layer1(x)   # 16 channels
        enc1 = self.layer2(enc0)  # 24
        enc2 = self.layer3(enc1)  # 40
        enc3 = self.layer4(enc2)  # 112
        enc4 = self.layer5(enc3)  # 960 (not 1280!)
        
        return self.bifpn({
            '0': enc0, '1': enc1, '2': enc2, 
            '3': enc3, '4': enc4
        })

def create_mobilenetv3_large_ssd(num_classes):
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32), (16, 32, 64), (32, 64, 128)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 3
    )
    
    head = SSDHead(
        in_channels=[128] * 3,
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )
    
    return SSD(
        backbone=MobileNetV3LargeBackbone(),
        anchor_generator=anchor_generator,
        head=head,
        num_classes=num_classes,
        size=(640, 640),
        iou_thresh=0.15,
        nms_thresh=0.25,
        score_thresh=0.01,
        detections_per_img=200
    )
    
if __name__ == '__main__':
    model = create_mobilenetv3_large_ssd(num_classes=20)
    print(model)