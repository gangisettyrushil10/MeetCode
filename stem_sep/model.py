import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down
        for f in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
            in_channels = f

        # Up
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels, f, kernel_size=2, stride=2))
            self.ups.append(nn.Sequential(
                nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU()
            ))
            in_channels = f

        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[-(idx//2 + 1)]
            x = torch.cat([x, skip], dim=1)
            x = self.ups[idx + 1](x)

        return torch.sigmoid(self.final(x))