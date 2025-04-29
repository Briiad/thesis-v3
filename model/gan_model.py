import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(SimpleGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output between [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(SimpleDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, 1, kernel_size=4, padding=1),
            nn.Sigmoid()  # Output patch-wise probability map
        )

    def forward(self, x):
        return self.net(x)
