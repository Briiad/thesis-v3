import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IMAGE_SIZE, NUM_CLASSES

class EnhancedLightCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(EnhancedLightCNN, self).__init__()
        
        # Block 1: 3 -> 64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        )
        
        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        
        # Block 3: 128 -> 256 (with increased capacity)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            nn.Conv2d(256, 256, 3, padding=1),  # Additional conv
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Block 4: 256 -> 512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = EnhancedLightCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output = model(dummy_input)
    print(output.shape)