import torch
from gatedconv import GatedConv


def soft_prune_step(network, prune_rate):
    for i in range(len(network.features)):
        if isinstance(network.features[i], GatedConv):
            kernel = network.features[i].conv.weight.data
            sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
            _, args = torch.sort(sum_of_kernel)
            soft_prune_list = args[:int(round(kernel.size(0) * prune_rate))].tolist()
            for j in soft_prune_list:
                network.features[i].conv.weight.data[j] = torch.zeros_like(network.features[i].conv.weight.data[j])
    return network
