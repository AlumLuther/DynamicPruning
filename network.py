import torch.nn as nn
import torch.nn.functional as F
import torch


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gate = nn.Linear(in_channels, out_channels)
        self.gate.weight = nn.init.kaiming_normal_(self.gate.weight)
        self.ratio = 0.5

    def forward(self, x):
        if 0:  # for test
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
            return x
        else:
            return self.gated_forward(x)

    def gated_forward(self, x):
        if self.ratio < 1:
            subsample = F.avg_pool2d(x, x.shape[2])
            subsample = subsample.view(x.shape[0], x.shape[1])
            gates = self.gate(subsample)
            gates = F.relu(gates)
            inactive_channels = self.conv.out_channels - round(self.conv.out_channels * self.ratio)
            inactive_idx = (-gates).topk(inactive_channels, 1)[1]
            gates.scatter_(1, inactive_idx, 0)

        x = self.conv(x)
        x = self.bn(x)
        if self.ratio < 1:
            x = x * gates.unsqueeze(2).unsqueeze(3)
        x = F.relu(x)

        return x


class MyVgg(nn.Module):
    def __init__(self):
        super(MyVgg, self).__init__()
        self.features = nn.Sequential(
            GatedConv(3, 64),
            GatedConv(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(64, 128),
            GatedConv(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(128, 256),
            GatedConv(256, 256),
            GatedConv(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(256, 512),
            GatedConv(512, 512),
            GatedConv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GatedConv(512, 512),
            GatedConv(512, 512),
            GatedConv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 10)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output
