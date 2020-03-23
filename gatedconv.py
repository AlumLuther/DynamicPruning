import torch.nn as nn
import torch.nn.functional as F


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gated=True, ratio=1):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gated = gated
        self.gate = nn.Linear(in_channels, out_channels)
        self.gate.weight = nn.init.kaiming_normal_(self.gate.weight)
        self.gate.bias = nn.init.constant_(self.gate.bias, 1)
        self.ratio = ratio

    def forward(self, x):
        if self.gated:
            subsample = F.avg_pool2d(x, x.shape[2])
            subsample = subsample.view(x.shape[0], x.shape[1])
            gates = self.gate(subsample)
            gates = F.relu(gates)
            if self.ratio < 1:
                inactive_channels = self.conv.out_channels - round(self.conv.out_channels * self.ratio)
                inactive_idx = (-gates).topk(inactive_channels, 1)[1]
                gates.scatter_(1, inactive_idx, 0)

        x = self.conv(x)
        x = self.bn(x)
        if self.gated:
            x = x * gates.unsqueeze(2).unsqueeze(3)
        x = F.relu(x)
        return x