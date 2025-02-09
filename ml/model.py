import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, IMAGE_SIZE

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(EfficientNetClassifier, self).__init__()
        # Load the pretrained EfficientNet-B0 model from torchvision
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Optionally, freeze the feature extractor:
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Replace the classifier head.
        # EfficientNet-B0 has a classifier like: Sequential(
        #   Dropout(p=0.2, inplace=True),
        #   Linear(in_features=1280, out_features=1000, bias=True)
        # )
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # For testing: create a dummy input and print the model summary.
    model = EfficientNetClassifier(num_classes=NUM_CLASSES)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output = model(dummy_input)
    print("Output shape:", output.shape)
