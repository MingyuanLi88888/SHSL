import torch.nn as nn
from dhg import Graph
import torch.nn.functional as F
from dhg.nn import HyperGCNConv, HGNNConv, UniSAGEConv, UniGINConv, UniGATConv
from torch.nn import Sequential, Linear, ReLU


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, proj_dim, dropout, args):

        super(GraphEncoder, self).__init__()

        self.hgnn_encoder_layers = nn.ModuleList()
        self.args = args

        if args.Encoder_type == 'HyperGCN':
            self.hgnn_encoder_layers.append(
                    HyperGCNConv(in_dim, hidden_dim, use_bn=False, drop_rate=dropout))
            self.hgnn_encoder_layers.append(
                    HyperGCNConv(hidden_dim, emb_dim, use_bn=False, is_last=True))
        elif args.Encoder_type == 'UniSAGE':
            self.hgnn_encoder_layers.append(
                    UniSAGEConv(in_dim, hidden_dim, use_bn=False, drop_rate=dropout))
            self.hgnn_encoder_layers.append(
                    UniSAGEConv(hidden_dim, emb_dim, use_bn=False, is_last=True))
        elif args.Encoder_type == 'UniGIN':
            self.hgnn_encoder_layers.append(
                    UniGINConv(in_dim, hidden_dim, use_bn=False, drop_rate=dropout))
            self.hgnn_encoder_layers.append(
                    UniGINConv(hidden_dim, emb_dim, use_bn=False, is_last=True))
        elif args.Encoder_type == 'HGNNP':
            self.hgnn_encoder_layers.append(
                    HGNNConv(in_dim, hidden_dim, use_bn=False, drop_rate=dropout))
            self.hgnn_encoder_layers.append(
                    HGNNConv(hidden_dim, emb_dim, use_bn=False, is_last=True))
        elif args.Encoder_type == 'UniGAT':
            self.hgnn_encoder_layers.append(
                    UniGATConv(in_dim, hidden_dim, use_bn=False, drop_rate=dropout))
            self.hgnn_encoder_layers.append(
                    UniGATConv(hidden_dim, emb_dim, use_bn=False, is_last=True))
        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))
        self.anchor_dropedge_rate = args.anchor_dropedge_rate
        self.hgyper_dropedge_rate = args.hgyper_dropedge_rate
    def forward(self,x, Adj_, branch=None):
        Adj_.to(x.device)
        if self.args.Encoder_type == 'HyperGCN':
            if branch == 'anchor':
                cached_g = Graph.from_hypergraph_hypergcn(Adj_, x, with_mediator=False)
                Adj_ = Adj_.drop_hyperedges(drop_rate=self.anchor_dropedge_rate)
                x = F.relu(self.hgnn_encoder_layers[0](x, Adj_, cached_g))
                x = self.hgnn_encoder_layers[1](x, Adj_, cached_g)
                z = self.proj_head(x)
                return z, x
            else:
                cached_g = Graph.from_hypergraph_hypergcn(Adj_, x, with_mediator=False)
                Adj_ = Adj_.drop_hyperedges(drop_rate=self.hgyper_dropedge_rate)
                x = F.relu(self.hgnn_encoder_layers[0](x, Adj_, cached_g))
                x = self.hgnn_encoder_layers[1](x, Adj_, cached_g)
                z = self.proj_head(x)
                return z, x
        else:
            if branch == 'anchor':
                Adj_ = Adj_.drop_hyperedges(drop_rate=self.anchor_dropedge_rate)
                x = F.relu(self.hgnn_encoder_layers[0](x, Adj_))
                x = self.hgnn_encoder_layers[1](x, Adj_)
                z = self.proj_head(x)
                return z, x
            else:
                Adj_ = Adj_.drop_hyperedges(drop_rate=self.hgyper_dropedge_rate)
                x = F.relu(self.hgnn_encoder_layers[0](x, Adj_))
                x = self.hgnn_encoder_layers[1](x, Adj_)
                z = self.proj_head(x)
                return z, x