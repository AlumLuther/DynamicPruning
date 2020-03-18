import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class DownSample(nn.Module):

    def __init__(self, stride):
        super(DownSample, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class GatedBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, down_sample=None, gated=False):
        super(GatedBlock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.down_sample = down_sample

        self.gated_flag_a = False
        self.gated_flag_b = gated
        self.gate_a = nn.Linear(inplanes, planes)
        self.gate_a.weight = nn.init.kaiming_normal_(self.gate_a.weight)
        self.gate_a.bias = nn.init.constant_(self.gate_a.bias, 1)
        self.gate_b = nn.Linear(planes, planes)
        self.gate_b.weight = nn.init.kaiming_normal_(self.gate_b.weight)
        self.gate_b.bias = nn.init.constant_(self.gate_b.bias, 1)
        self.ratio = 1

    def forward(self, x):
        residual = x

        if self.gated_flag_a:
            subsample = F.avg_pool2d(x, x.shape[2])
            subsample = subsample.view(x.shape[0], x.shape[1])
            gates = self.gate_a(subsample)
            gates = F.relu(gates)
            if self.ratio < 1:
                inactive_channels = self.conv_a.out_channels - round(self.conv_a.out_channels * self.ratio)
                inactive_idx = (-gates).topk(inactive_channels, 1)[1]
                gates.scatter_(1, inactive_idx, 0)

        x = self.conv_a(x)
        x = self.bn_a(x)
        if self.gated_flag_a:
            x = x * gates.unsqueeze(2).unsqueeze(3)
        x = F.relu(x, inplace=True)

        if self.gated_flag_b:
            subsample = F.avg_pool2d(x, x.shape[2])
            subsample = subsample.view(x.shape[0], x.shape[1])
            gates = self.gate_b(subsample)
            gates = F.relu(gates)
            if self.ratio < 1:
                inactive_channels = self.conv_b.out_channels - round(self.conv_b.out_channels * self.ratio)
                inactive_idx = (-gates).topk(inactive_channels, 1)[1]
                gates.scatter_(1, inactive_idx, 0)

        x = self.conv_b(x)
        x = self.bn_b(x)
        if self.gated_flag_b:
            x = x * gates.unsqueeze(2).unsqueeze(3)
        if self.down_sample is not None:
            residual = self.down_sample(residual)
        x = F.relu(x + residual, inplace=True)
        return x


class CifarResNet(nn.Module):
    def __init__(self, block, depth, num_classes, cfg=[16, 32, 64]):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert depth in [20, 32, 44, 56, 110]
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(cfg[0])

        self.inplanes = cfg[0]
        self.stage_1 = self._make_layer(block, cfg[0], layer_blocks, 1, gated=False)
        self.stage_2 = self._make_layer(block, cfg[1], layer_blocks, 2, gated=False)
        self.stage_3 = self._make_layer(block, cfg[2], layer_blocks, 2, gated=True)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(cfg[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, gated=False):
        down_sample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample = DownSample(stride)

        layers = [block(self.inplanes, planes, stride, down_sample, gated=gated)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, gated=gated))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(GatedBlock, 20, num_classes)
    return model


def resnet32(num_classes=10, cfg=[16, 32, 64]):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(GatedBlock, 32, num_classes, cfg)
    return model


def resnet44(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(GatedBlock, 44, num_classes)
    return model


def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(GatedBlock, 56, num_classes)
    return model


def resnet110(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(GatedBlock, 110, num_classes)
    return model
