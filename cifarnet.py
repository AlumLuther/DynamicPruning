import torch.nn as nn
from gatedconv import GatedConv


class CifarNet(nn.Module):
    def __init__(self, num_class=10, gated=False, ratio=1):
        super(CifarNet, self).__init__()
        self.features = nn.Sequential(
            GatedConv(3, 64, padding=0, gated=gated, ratio=ratio),
            GatedConv(64, 64, gated=gated, ratio=ratio),
            GatedConv(64, 128, stride=2, gated=gated, ratio=ratio),
            GatedConv(128, 128, gated=gated, ratio=ratio),
            GatedConv(128, 128, gated=gated, ratio=ratio),
            GatedConv(128, 192, stride=2, gated=gated, ratio=ratio),
            GatedConv(192, 192, gated=gated, ratio=ratio),
            GatedConv(192, 192, gated=gated, ratio=ratio),
            nn.AvgPool2d(8)
        )
        self.fc = nn.Linear(192, num_class)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 192)
        x = self.fc(x)

        return x
