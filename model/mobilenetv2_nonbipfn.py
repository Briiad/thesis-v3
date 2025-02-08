import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator

class MobileNetV2SSDBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        # Split the backbone into three stages:
        # stage1: features[:7]   (output channels ~32)
        # stage2: features[7:14] (output channels ~320)
        # stage3: features[14:]  (output channels ~1280)
        self.stage1 = nn.Sequential(*backbone[:7])
        self.stage2 = nn.Sequential(*backbone[7:18])  # Instead of 7:14
        self.stage3 = nn.Sequential(*backbone[18:])
        # Convert each stage’s output to 256 channels (instead of 128)
        self.conv1x1_1 = nn.Conv2d(32, 256, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(320, 256, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        # Additional convolution blocks to further increase parameters
        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.extra_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.stage1(x)
        print("f1 shape:", f1.shape)  # Should be [B, 32, H, W]
        f2 = self.stage2(f1)
        print("f2 shape:", f2.shape)  # Should be [B, 320, H, W], but it's 96
        f3 = self.stage3(f2)
        print("f3 shape:", f3.shape)  # Should be [B, 1280, H, W]
        p1 = self.conv1x1_1(f1)
        p2 = self.conv1x1_2(f2)  # ERROR happens here!
        p3 = self.conv1x1_3(f3)
        return {'0': p1, '1': p2, '2': p3}


def create_mobilenetv2_ssd(num_classes):
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32), (16, 32, 64), (32, 64, 128)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 3
    )
    head = SSDHead(
        in_channels=[256, 256, 256],  # Updated to match the 256 channels
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )
    return SSD(
        backbone=MobileNetV2SSDBackbone(),
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
    model = create_mobilenetv2_ssd(num_classes=20)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.eval()
    image = torch.randn(1, 3, 640, 640)
    output = model(image)
    print("Output shapes:", {k: v.shape for k, v in output[0].items()})
