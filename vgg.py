import torch.nn as nn
from gatedconv import GatedConv


class MyVgg(nn.Module):
    def __init__(self, gated=True, ratio=1):
        super(MyVgg, self).__init__()
        self.features = nn.Sequential(
            GatedConv(3, 64),
            GatedConv(64, 64, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(64, 128),
            GatedConv(128, 128, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(128, 256),
            GatedConv(256, 256, gated=gated, ratio=ratio),
            GatedConv(256, 256, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(256, 512),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            GatedConv(512, 512, gated=gated, ratio=ratio),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output
