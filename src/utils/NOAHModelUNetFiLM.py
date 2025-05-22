
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, preds, targets):
        # torchmetrics returns similarity (higher is better), so we invert it for loss
        ssim_score = self.ssim(preds, targets)
        return 1 - ssim_score

class FiLM(nn.Module):
    def __init__(self, num_features, conditioning_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(conditioning_dim, num_features)
        self.beta_fc = nn.Linear(conditioning_dim, num_features)

    def forward(self, x, cond):
        # x: [B, C, H, W], cond: [B, F]
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return x * gamma + beta

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DownBlock(in_channels, out_channels)
        self.film = FiLM(out_channels, conditioning_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.film(x, cond)
        return x

class UNetFiLM(nn.Module):
    def __init__(self, in_channels, conditioning_dim, out_channels=1):
        super().__init__()
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DownBlock(256, 512)

        self.up3 = UpBlock(512, 256, conditioning_dim)
        self.up2 = UpBlock(256, 128, conditioning_dim)
        self.up1 = UpBlock(128, 64, conditioning_dim)

        # this specifically need to be there becaused the dim is 1333 and it need to be divisible by 8
        self.padding = (1, 2, 1, 2)  # (left, right, top, bottom)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x_img, x_weather):
        # get (B, 5, 1333, 1333) -> (B, 5, 1336, 1336)
        x_img = pad(x_img, self.padding, mode='constant', value=0)
        
        e1 = self.enc1(x_img)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b, e3, x_weather)
        d2 = self.up2(d3, e2, x_weather)
        d1 = self.up1(d2, e1, x_weather)

        out = self.final(d1)
        # crop so as to get input dim
        out = out[:, :, 1:-2, 1:-2]
        return out
