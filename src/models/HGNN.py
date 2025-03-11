import torch
import torch.nn as nn
from dhg.nn import HGNNConv
class HGNN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        hypergraph = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )
        self.hypergraph = hypergraph
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.hypergraph.to(X.device)
        for layer in self.layers:
            X = layer(X, self.hypergraph)
        return X
