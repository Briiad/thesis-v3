import torch
import torch.nn as nn
import torchvision.models as models
from ml.config import NUM_CLASSES, IMAGE_SIZE

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
      super(EfficientNetClassifier, self).__init__()
      # Load the pretrained EfficientNet-B0 model
      self.model = models.efficientnet_b0(pretrained=True)

      # Freeze all layers except the last feature block to allow fine-tuning
      for name, param in self.model.named_parameters():
        # Unfreeze the last block (commonly "features.6") and the classifier head
        if "features.6" in name:
          param.requires_grad = True
        else:
          param.requires_grad = False

      # Replace the classifier head with a higher dropout rate for added regularization
      in_features = self.model.classifier[1].in_features
      self.model.classifier = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(in_features, num_classes)
      )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # For testing: create a dummy input and print the model summary.
    model = EfficientNetClassifier(num_classes=NUM_CLASSES)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output = model(dummy_input)
    print("Output shape:", output.shape)