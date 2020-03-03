import torch

from parameter import get_parameter
from train import train_network
from utils import save_network, load_network

torch.cuda.set_device(0)
args = get_parameter()
model = load_network(args)
model = train_network(model, args)
save_network(model, args)