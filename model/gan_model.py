# model/gan_generator.py

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(UNetGenerator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Bottleneck
        self.bot = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bot(e3)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)

class UNetDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(UNetDiscriminator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Bottleneck
        self.bot = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Decoder
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Output (patch probability map)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bot(e3)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)
