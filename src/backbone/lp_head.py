import torch
import torch.nn as nn


class LinkPredHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        return torch.sum(x_src * x_dst, dim=1)
    