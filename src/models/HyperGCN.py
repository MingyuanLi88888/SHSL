import torch
import torch.nn as nn

import dhg
from dhg.structure.graphs import Graph
from typing import Optional

import torch
import torch.nn as nn

from dhg.structure.graphs import Graph
from dhg.structure.hypergraphs import Hypergraph


class HyperGCNConv(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mediator: bool = False,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,

    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.use_mediator = use_mediator
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
    def forward(
        self, X: torch.Tensor, hg: Hypergraph, cached_g: Optional[Graph] = None
    ) -> torch.Tensor:

        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        if cached_g is None:
            g = Graph.from_hypergraph_hypergcn(
                hg, X, self.use_mediator, device=X.device
            )
            X = g.smoothing_with_GCN(X)
        else:
            X = cached_g.smoothing_with_GCN(X)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X

class HyperGCN(nn.Module):

    def __init__(
            self,
            in_channels: int,
            hid_channels: int,
            num_classes: int,
            use_mediator: bool = False,
            use_bn: bool = False,
            fast: bool = True,
            drop_rate: float = 0.5,
            hypergraph=None,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.layers = nn.ModuleList()
        self.layers.append(
            HyperGCNConv(
                in_channels, hid_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate,
            )
        )
        self.layers.append(
            HyperGCNConv(
                hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
            )
        )
        self.hypergraph = hypergraph

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        if self.fast:
            if self.cached_g is None:
                self.cached_g = Graph.from_hypergraph_hypergcn(
                    self.hypergraph, X, self.with_mediator
                )
            for layer in self.layers:
                X = layer(X, self.hypergraph, self.cached_g)
        else:
            for layer in self.layers:
                X = layer(X, self.hypergraph)
        return X
