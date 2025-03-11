import torch
import torch.nn as nn
import dhg
from dhg.nn import UniGATConv, MultiHeadWrapper


class UniGAT(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_heads: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        hypergraph = None,
    ) -> None:
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            UniGATConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )

        self.out_layer = UniGATConv(
            hid_channels * num_heads,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )
        self.hypergraph = hypergraph
    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X = self.drop_layer(X)
        X = self.multi_head_layer(X=X, hg=self.hypergraph)
        X = self.drop_layer(X)
        X = self.out_layer(X, self.hypergraph)
        return X