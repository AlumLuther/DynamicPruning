import torch
import torchvision
import torchvision.transforms as transforms

from cifarnet import CifarNet
from resnet import resnet20, resnet32
from vgg import MyVgg


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return transforms.Normalize(MEAN, STD)


def get_transformer(data_set, imsize=None, cropsize=None, crop_padding=None, hflip=None):
    transformers = []
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        transformers.append(transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip(hflip))

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))

    return transforms.Compose(transformers)


def get_data_set(args, train_flag=True):
    if train_flag:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True,
                                                                transform=get_transformer(args.data_set, args.imsize,
                                                                                          args.cropsize, args.crop_padding, args.hflip), download=True)
    else:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False,
                                                                transform=get_transformer(args.data_set), download=True)
    return data_set


def load_network(args):
    network = None
    if args.load_path:
        check_point = torch.load(args.load_path)
        if args.network == 'vgg':
            network = MyVgg(gated=args.gated, ratio=args.ratio)
            network.load_state_dict(check_point['state_dict'])
        elif args.network == 'resnet':
            network = resnet32(gated=args.gated, ratio=args.ratio)
            network.load_state_dict(check_point['state_dict'])
        elif args.network == 'cifarnet':
            network = CifarNet(gated=args.gated, ratio=args.ratio)
            network.load_state_dict(check_point['state_dict'])
    else:
        if args.network == 'vgg':
            network = MyVgg(gated=args.gated, ratio=args.ratio)
        elif args.network == 'resnet':
            network = resnet32(gated=args.gated, ratio=args.ratio)
        elif args.network == 'cifarnet':
            network = CifarNet(gated=args.gated, ratio=args.ratio)
    return network


def save_network(network, args):
    if args.save_path:
        torch.save({'state_dict': network.state_dict()}, args.save_path)
