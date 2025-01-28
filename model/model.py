import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

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

    return SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        head=head,
        iou_thresh=0.2
    )

if __name__ == '__main__':
    model = create_ssd_model(num_classes=7)
    model.eval()
    image = torch.randn(1, 3, 640, 640)
    output = model(image)
    print("Output shapes:", {k: v.shape for k, v in output[0].items()})
  