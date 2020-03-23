import torch.nn as nn
from gatedconv import GatedConv


class CifarNet(nn.Module):
    def __init__(self, gated=True, ratio=1):
        super(CifarNet, self).__init__()
        self.gconv0 = GatedConv(3, 64, padding=0, gated=gated, ratio=ratio)
        self.gconv1 = GatedConv(64, 64, gated=gated, ratio=ratio)
        self.gconv2 = GatedConv(64, 128, stride=2, gated=gated, ratio=ratio)
        self.gconv3 = GatedConv(128, 128, gated=gated, ratio=ratio)
        self.drop3 = nn.Dropout2d()
        self.gconv4 = GatedConv(128, 128, gated=gated, ratio=ratio)
        self.gconv5 = GatedConv(128, 192, stride=2, gated=gated, ratio=ratio)
        self.gconv6 = GatedConv(192, 192, gated=gated, ratio=ratio)
        self.drop6 = nn.Dropout2d()
        self.gconv7 = GatedConv(192, 192, gated=gated, ratio=ratio)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.gconv0(x)
        x = self.gconv1(x)
        x = self.gconv2(x)
        x = self.gconv3(x)
        x = self.drop3(x)
        x = self.gconv4(x)
        x = self.gconv5(x)
        x = self.gconv6(x)
        x = self.drop6(x)
        x = self.gconv7(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)

        return x
