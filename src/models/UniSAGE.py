import torch
import torch.nn as nn
from dhg.nn import UniSAGEConv

from dhg.structure.hypergraphs import Hypergraph
class UniSAGE(nn.Module):
    def __init__(
        self, in_channels: int, hid_channels: int, num_classes: int, hypergraph, use_bn: bool = False, drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UniSAGEConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(UniSAGEConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))
        self.hypergraph = hypergraph
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, self.hypergraph)
        return X