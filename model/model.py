import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class CustomBackboneWithFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
        features = backbone.features
        
        # Corrected layer splits with verified output channels
        self.layer1 = nn.Sequential(*features[0:2])   # Output: 16 channels
        self.layer2 = nn.Sequential(*features[2:4])   # Output: 24 channels
        self.layer3 = nn.Sequential(*features[4:7])   # Output: 32 channels
        self.layer4 = nn.Sequential(*features[7:14])  # Output: 96 channels (not 64)
        self.layer5 = nn.Sequential(*features[14:])   # Output: 1280 channels
        
        # Update FPN input channels to match actual backbone outputs
        in_channels_list = [16, 24, 32, 96, 1280]  # Adjusted for layer4's 96 channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=1280
        )

    def forward(self, x):
        enc0 = self.layer1(x)    # [16, H, W]
        enc1 = self.layer2(enc0) # [24, H/2, W/2]
        enc2 = self.layer3(enc1) # [32, H/4, W/4]
        enc3 = self.layer4(enc2) # [96, H/8, W/8] (previously incorrect as 64)
        enc4 = self.layer5(enc3) # [1280, H/16, W/16]

        feature_maps = {'0': enc0, '1': enc1, '2': enc2, '3': enc3, '4': enc4}
        return self.fpn(feature_maps)

def create_ssd_model(num_classes, pretrained_backbone=True):
    backbone = CustomBackboneWithFPN(pretrained=pretrained_backbone)
    
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256), (128, 256, 512)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    head = SSDHead(
        in_channels=[1280] * 5,  # FPN outputs all have 1280 channels
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )

    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(320, 320),
        head=head
    )
    return model
  
if __name__ == '__main__':
    model = create_ssd_model(num_classes=7)
    model.eval() 
    
    # Test the model
    image = torch.randn(1, 3, 320, 320)
    output = model(image)
    print("Output shape:", output)