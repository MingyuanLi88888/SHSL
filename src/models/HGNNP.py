import torch
import torch.nn as nn

import dhg
from dhg.nn import HGNNPConv


class HGNNP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        hypergraph=None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )
        self.hypergraph = hypergraph
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, self.hypergraph)
        return X
