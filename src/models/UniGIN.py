import torch
import torch.nn as nn
from dhg.nn import UniSAGEConv, UniGINConv


class UniGIN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        hypergraph:None,
        eps: float = 0.0,
        train_eps: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            UniGINConv(in_channels, hid_channels, eps=eps, train_eps=train_eps, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            UniGINConv(hid_channels, num_classes, eps=eps, train_eps=train_eps, use_bn=use_bn, is_last=True)
        )
        self.hypergraph = hypergraph
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, self.hypergraph)
        return X
