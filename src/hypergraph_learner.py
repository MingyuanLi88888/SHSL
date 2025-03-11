import dhg
import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph
from sklearn.cluster import KMeans
import numpy as np




def initialize_hypergraph(features, fea_dim, num_clusters, args):
    # features = torch.tensor(features)
    # features = F.normalize(features, dim=1, p=2)
    # normalized_embeddings = features / torch.norm(features, dim=1, keepdim=True)
    # similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
    # similarities = F.normalize(similarities, dim=1, p=2)
    # similarities = similarities / torch.norm(similarities, dim=1, keepdim=True)
    # S_sum = torch.sum(similarities, dim=1)
    # D = torch.diag(S_sum)
    # L = D - similarities
    # eigvals, eigvecs = torch.symeig(L, eigenvectors=True)
    # clustering_input = eigvecs[:, :fea_dim]
    # HG = dhg.load_structure('hypergraph_struct/CoauthorshipCora.hg')
    print(args)
    if args.dataset == 'CocitationCora':
        H = dhg.load_structure('../hypergraph_struct/CoauthorshipCora.hg')
    if args.dataset == 'CoauthorshipCora':
        H = dhg.load_structure('../hypergraph_struct/CoauthorshipCora.hg')
    if args.dataset == 'CoauthorshipDBLP':
        H = dhg.load_structure('../hypergraph_struct/CoauthorshipDBLP.hg')
    if args.dataset == 'CocitationCiteseer':
        H = dhg.load_structure('../hypergraph_struct/CocitationCiteseer.hg')
    if args.dataset == 'CocitationPubmed':
        H = dhg.load_structure('../hypergraph_struct/CocitationPubmed.hg')
    if args.dataset == 'News20':
        H = dhg.load_structure('../hypergraph_struct/News20.hg')
    if args.dataset == 'diabetes':
        H = dhg.load_structure('../hypergraph_struct/diabetes.hg')
    if args.dataset == 'breast_cancer':
        H = dhg.load_structure('../hypergraph_struct/breast_cancer.hg')
    return H.H.to_dense()
class Attentive(nn.Module):
    def __init__(self, insize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(insize))
    def forward(self, x):
        return x @ torch.diag(self.w)
class iecs_hgraph_learner(nn.Module):
    def __init__(self, features, fea_dim, num_clusters, args):
        super(iecs_hgraph_learner, self).__init__()
        self.Adj = nn.Parameter(
            initialize_hypergraph(features, fea_dim, num_clusters, args))
    def forward(self, h):
        Adj = F.elu(self.Adj) + 1
        return Adj
    
def cal_similarity_graph(node_embeddings):
    normalized_embeddings = node_embeddings / torch.norm(node_embeddings, dim=1, keepdim=True)
    similarity_graph = torch.mm(normalized_embeddings, normalized_embeddings.t())
    return similarity_graph
class att_hgraph_learner(nn.Module):
    def __init__(self, features, fea_dim, num_clusters, args):
        super(att_hgraph_learner, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            self.layers.append(Attentive(features.shape[1]))
        self.mlp_act = 'relu'
        self.num_clusters = num_clusters
    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h
    def forward(self, h):
        embeddings = self.internal_forward(h)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = cal_similarity_graph(embeddings)
        H = Hypergraph.from_feature_kNN(similarities.detach().cpu(), self.num_clusters)
        return H.H.to_dense()