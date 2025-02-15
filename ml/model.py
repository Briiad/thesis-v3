import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, IMAGE_SIZE

class MobileNetV3Classifier(nn.Module):
  def __init__(self, num_classes=NUM_CLASSES, freeze_feature_extractor=False):
    super(MobileNetV3Classifier, self).__init__()
    # Load the pretrained MobileNetV3 Large model
    self.model = models.mobilenet_v3_large(pretrained=True)
    
    if freeze_feature_extractor:
      for param in self.model.parameters():
        param.requires_grad = False

    # Replace the classifier head.
    # MobileNetV3 Large's classifier is a Sequential, with the last layer being the final Linear layer.
    in_features = self.model.classifier[-1].in_features
    self.model.classifier[-1] = nn.Linear(in_features, num_classes)

  def forward(self, x):
    return self.model(x)

if __name__ == '__main__':
  # For testing: create a dummy input and print the model summary.
  model = MobileNetV3Classifier(num_classes=NUM_CLASSES)
  print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
  dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
  output = model(dummy_input)
  print("Output shape:", output.shape)
