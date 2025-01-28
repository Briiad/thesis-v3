import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import cv2
import numpy as np
from torchvision import transforms

class CustomBackboneWithFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load MobileNetV2 as backbone
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
        features = backbone.features
        self.layer1= nn.Sequential(*features[0:4])
        self.layer2 = nn.Sequential(*features[4:7])
        self.layer3 = nn.Sequential(*features[7:11])
        self.layer4 = nn.Sequential(*features[11:19])
        for param in features.parameters():
            param.requires_grad = False
        
        # Last output channel from convolutional layers from MobileNetV2 model
        self.backbone_out_channels =  1280
        
        # Define FPN
        in_channels_list = [24, 32, 96, 320] 
        self.out_channels = [1280, 1280, 1280, 1280, 1280]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=1280
        )

    def forward(self, x):
        enc0 = self.layer1(x) # 24
        enc1 = self.layer2(enc0) # 32
        enc2 = self.layer3(enc1) # 64
        enc3 = self.layer4(enc2) # 1280
        return enc3


def create_ssd_model(num_classes, pretrained_backbone=True):
    # Create custom backbone with FPN
    backbone = CustomBackboneWithFPN(pretrained=pretrained_backbone)
    
    # Define anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
      
    num_anchors = anchor_generator.num_anchors_per_location()
    head = SSDHead(backbone.out_channels, num_anchors, num_classes)

    # Define the SSD model using torchvision's implementation
    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(640, 640),
        head=head
    )
    
    return model

# # Example usage
# if __name__ == "__main__":
  
#   # Create model with 91 classes (COCO dataset)
#   model = create_ssd_model(num_classes=91)
#   model.eval()
  
#   # Test with random tensor
#   print("Testing with random tensor:")
#   x_random = torch.rand(2, 3, 320, 320)
#   predictions_random = model(x_random)
  
#   for k, v in predictions_random[0].items():
#     print(f"{k}: shape {v.shape}")
  
#   # Test with actual image
#   print("\nTesting with sample image:")
#   # Create a sample image with some shapes
#   img = np.zeros((320, 320, 3), dtype=np.uint8)
#   cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Draw a blue rectangle
#   cv2.circle(img, (250, 250), 30, (0, 255, 0), -1)  # Draw a green circle
  
#   # Transform image for model
#   transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                std=[0.229, 0.224, 0.225])
#   ])
  
#   img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
  
#   # Get predictions
#   with torch.no_grad():
#     predictions_img = model(img_tensor)
  
#   print("\nPredictions for sample image:")
#   for k, v in predictions_img[0].items():
#     print(f"{k}: shape {v.shape}")
#     if k == 'scores':
#       print(f"Top 3 confidence scores: {v[:3]}")