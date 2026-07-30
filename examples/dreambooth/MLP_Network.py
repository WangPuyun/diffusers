import torch.nn as nn


class MLP(nn.Module):
    """Predict a three-channel image at 16 times the input spatial resolution using linear layers."""

    def __init__(self, in_dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(in_dim * mlp_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, 3 * 16**2)
        self.drop2 = nn.Dropout(drop)
        self.pixel_shuffle = nn.PixelShuffle(16)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.permute(0, 3, 1, 2)
        return self.pixel_shuffle(x)


class ConvMLP(nn.Module):
    """Map a BCHW feature map to a three-channel image at 16 times the input spatial resolution."""

    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
