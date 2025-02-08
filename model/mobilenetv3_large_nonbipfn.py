import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator

class MobileNetV3LargeSSDBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
        # Group layers as in MobileNetV3-Large:
        # layer1: backbone[:2]    (output ~16 channels; not used for detection here)
        # layer2: backbone[2:4]   (output ~24 channels; not used for detection)
        # layer3: backbone[4:7]   (output ~40 channels)  -> detection feature 1
        # layer4: backbone[7:13]  (output ~112 channels) -> detection feature 2
        # layer5: backbone[13:]   (output ~960 channels) -> detection feature 3
        self.layer1 = nn.Sequential(*backbone[:2])
        self.layer2 = nn.Sequential(*backbone[2:4])
        self.layer3 = nn.Sequential(*backbone[4:7])   # 40 channels
        self.layer4 = nn.Sequential(*backbone[7:13])  # 112 channels
        self.layer5 = nn.Sequential(*backbone[13:])   # 960 channels
        # Convert to 256 channels per feature map
        self.conv1x1_1 = nn.Conv2d(40, 256, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(112, 256, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(960, 256, kernel_size=1)
        # Additional convolution blocks to increase parameter count
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
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)    # Feature map from layer3 (40 channels)
        f4 = self.layer4(f3)   # Feature map from layer4 (112 channels)
        f5 = self.layer5(f4)   # Feature map from layer5 (960 channels)
        p1 = self.conv1x1_1(f3)
        p2 = self.conv1x1_2(f4)
        p3 = self.conv1x1_3(f5)
        p1 = self.extra_conv1(p1)
        p2 = self.extra_conv2(p2)
        p3 = self.extra_conv3(p3)
        return {'0': p1, '1': p2, '2': p3}

def create_mobilenetv3_large_ssd(num_classes):
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
        backbone=MobileNetV3LargeSSDBackbone(),
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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.eval()
    image = torch.randn(1, 3, 640, 640)
    output = model(image)
    print("Output shapes:", {k: v.shape for k, v in output[0].items()})