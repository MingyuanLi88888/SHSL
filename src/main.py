import copy
import dhg
import torch
import numpy as np
import dgl
import warnings
import random

from dhg.data import CocitationCora, Cooking200, CoauthorshipCora, CoauthorshipDBLP, \
    CocitationCiteseer, CocitationPubmed, HouseCommittees, News20
from dhg import Hypergraph
from sklearn.cluster import KMeans
import torch.nn.functional as F
from src import config
from GCL_2 import GCL
from src.hypergraph_learner import iecs_hgraph_learner, att_hgraph_learner
from src.models.HGNNP import HGNNP
from src.models.HyperGCN import HyperGCN
from src.models.UniGAT import UniGAT
from src.models.UniGIN import UniGIN
from src.models.UniSAGE import UniSAGE
from src.utils import clustering_metrics, print_results, toHypergraph, read_diabetes, read_breast_cancer

warnings.filterwarnings("ignore")
def huafen(lbl):
    train_ratio = 0.8
    val_ratio = 0.1
    total_samples = len(lbl)
    indices = torch.randperm(total_samples)  # 随机排列索引
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    train_mask = torch.zeros(total_samples, dtype=torch.bool)
    val_mask = torch.zeros(total_samples, dtype=torch.bool)
    test_mask = torch.zeros(total_samples, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    return train_mask, val_mask, test_mask

def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]

def loss_cls(model, mask, features, labels):
    logits = model(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
    accu = accuracy(logp[mask], labels[mask])
    return loss, accu

def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda(), samples
def loss_gcl(model, hgraph_learner, features, anchor_adj):

    # view 1: anchor graph
    if args.anchor_featuremask:
        mask_v1, _ = get_feat_mask(features, args.anchor_featuremask)
        features_v1 = features * (1 - mask_v1)
    else:
        features_v1 = copy.deepcopy(features)
    z1, _ = model(features_v1, anchor_adj, branch='anchor')
    # view 2: hypergraph_learned
    if args.hgraphlearner_featuremask:
        mask, _ = get_feat_mask(features, args.hgraphlearner_featuremask)
        features_v2 = features * (1 - mask)
    else:
        features_v2 = copy.deepcopy(features)

    learned_adj = hgraph_learner(features)
    learned_adj = toHypergraph(learned_adj)
    z2, _ = model(features_v2, learned_adj, 'learner')
    # compute loss
    loss = model.calc_loss(z1, z2, args.tau)
    return loss, learned_adj, z1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
def read_data(args):
    if args.dataset == 'CocitationCora':
        data = CocitationCora()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('../hypergraph_struct/CocitationCora.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    elif args.dataset == 'CoauthorshipCora':
        data = CoauthorshipCora()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('../hypergraph_struct/CoauthorshipCora.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    elif args.dataset == 'CoauthorshipDBLP':
        data = CoauthorshipDBLP()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('../hypergraph_struct/CoauthorshipDBLP.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    elif args.dataset == 'CocitationPubmed':
        data = CocitationPubmed()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('../hypergraph_struct/CocitationPubmed.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    elif args.dataset == 'CocitationCiteseer':
        data = CocitationCiteseer()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('../hypergraph_struct/CocitationCiteseer.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    elif args.dataset == 'HouseCommittees':
        data = HouseCommittees()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('hypergraph_struct/HouseCommittees.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    elif args.dataset == 'Cooking200':
        data = Cooking200()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('hypergraph_struct/Cooking200.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    elif args.dataset == 'diabetes':
        X, lbl = read_diabetes()
        HG = dhg.load_structure('../hypergraph_struct/diabetes.hg')
        train_mask, val_mask, test_mask = huafen(lbl)
        nclasses = len(torch.unique(lbl))
    elif args.dataset == 'breast_cancer':
        X, lbl = read_breast_cancer()
        HG = dhg.load_structure('../hypergraph_struct/breast_cancer.hg')
        train_mask, val_mask, test_mask = huafen(lbl)
        nclasses = len(torch.unique(lbl))
    elif args.dataset == 'News20':
        data = News20()
        if args.gsl_mode == 'structure_inference':
            HG = dhg.load_structure('../hypergraph_struct/News20.hg')
        if args.gsl_mode == 'structure_refinement':
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
        train_mask, val_mask, test_mask = huafen(data["labels"])
        nclasses = data["num_classes"]
        X, lbl = data["features"], data["labels"]
    else:
        data = None
        print('Error Dataset')
    features = X
    features = F.normalize(features, dim=1, p=2)
    features = features / torch.norm(features, dim=1, keepdim=True)
    labels = lbl
    nfeats = features.shape[1]

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, HG

def evaluate_hyper_link():


    print('name ')
def evaluate_adj_by_cls(hypergraph, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):
    if args.Encoder_type == 'HyperGCN':
        model = HyperGCN(in_channels=nfeats, hid_channels=args.hidden_dim_classifier, num_classes=nclasses, drop_rate=args.dropout_classifier, hypergraph=hypergraph)
    elif args.Encoder_type == 'UniSAGE':
        model = UniSAGE(in_channels=nfeats, hid_channels=args.hidden_dim_classifier, num_classes=nclasses, drop_rate=args.dropout_classifier, hypergraph=hypergraph)
    elif args.Encoder_type == 'UniGIN':
        model = UniGIN(in_channels=nfeats, hid_channels=args.hidden_dim_classifier, num_classes=nclasses, drop_rate=args.dropout_classifier, hypergraph=hypergraph)
    elif args.Encoder_type == 'HGNNP':
        model = HGNNP(in_channels=nfeats, hid_channels=args.hidden_dim_classifier, num_classes=nclasses, drop_rate=args.dropout_classifier, hypergraph=hypergraph)
    elif args.Encoder_type == 'UniGAT':
        model = UniGAT(in_channels=nfeats, hid_channels=args.hidden_dim_classifier, num_classes=nclasses, drop_rate=args.dropout_classifier, hypergraph=hypergraph, num_heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_classifier, weight_decay=args.wd_classifier)
    bad_counter = 0
    best_val = 0
    best_model = None

    if torch.cuda.is_available():
        model = model.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        features = features.cuda()
        labels = labels.cuda()
    for epoch in range(1, args.epochs_classifier + 1):
        model.train()
        loss, accu = loss_cls(model, train_mask, features, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            val_loss, accu = loss_cls(model, val_mask, features, labels)
            if accu >= best_val:
                bad_counter = 0
                best_val = accu
                best_model = copy.deepcopy(model)
            else:
                bad_counter += 1
            if bad_counter >= args.patience:
                break
    best_model.eval()
    test_loss, test_accu = loss_cls(best_model, test_mask, features, labels)
    return best_val, test_accu, best_model


def get_hyper_features(features, edge_index):
    edge_feats = []
    for i in edge_index:
        edge_feats.append(torch.mean(features[i], dim=0))
    edge_feats = torch.stack(edge_feats)
    return edge_feats


def main(args):
    if args.gsl_mode == 'structure_inference':
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = read_data(args)
        anchor_adj = Hypergraph.from_feature_kNN(features, args.num_clusters)
    if args.gsl_mode == 'structure_refinement':
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, HG = read_data(args)
        anchor_adj = HG
        # neg_edge = neg_sampler(HG, args.dataset)
        # print(neg_edge)
    for trial in range(args.ntrials):
        Train_loss = []
        setup_seed(trial)
        if args.hypergraph_learner == 'iecs':
            hypergraph_learner = iecs_hgraph_learner(features, args.fea_dim, args.num_clusters, args)
        if args.hypergraph_learner == 'att':
            hypergraph_learner = att_hgraph_learner(features, args.fea_dim, args.num_clusters, args)
        model = GCL(input_dim=nfeats, hidden_dim=args.hidden_dim,
                    embedding_dim=args.embedding_dim, proj_dim=args.proj_dim,
                    dropout=args.dropout, args=args)
        encoder_optimizer = torch.optim.Adam(model.parameters(), lr=args.encoder_lr, weight_decay=args.encoder_wd)
        hyperlearner_optimizer = torch.optim.Adam(hypergraph_learner.parameters(), lr=args.encoder_lr, weight_decay=args.encoder_wd)


        if torch.cuda.is_available():
            model = model.cuda()
            hypergraph_learner = hypergraph_learner.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()
            anchor_adj = anchor_adj.to(features.device)
        if args.downstream_task == 'multi-task':
            test_accuracies = []
            validation_accuracies = []
            best_test = 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                hypergraph_learner.train()
                loss, hypergraph, z = loss_gcl(model, hypergraph_learner, features, anchor_adj)
                encoder_optimizer.zero_grad()
                hyperlearner_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                hyperlearner_optimizer.step()
                if epoch % args.eval_freq == 0:
                    model.eval()
                    hypergraph_learner.eval()
                    val_accu, test_accu, _ = evaluate_adj_by_cls(anchor_adj, features, nfeats, labels,
                                                                 nclasses, train_mask, val_mask, test_mask, args)

                    # if args.gsl_mode == 'structure_refinement':
                    #     neg_hyper_feature = get_hyper_features(z, neg_edge)
                    #     pos_list = [list(subset) for subset in anchor_adj.e[0]]
                    #     pos_hyper_feature = get_hyper_features(z, pos_list)
                    #     y_pos = torch.ones(pos_hyper_feature.shape[0], dtype=torch.float32)
                    #     y_neg = torch.zeros(neg_hyper_feature.shape[0], dtype=torch.float32)
                    #     y = torch.concat([y_pos, y_neg], dim=0)
                    #     x = torch.cat([pos_hyper_feature, neg_hyper_feature], dim=0)
                    #     split = get_split(num_samples=x.size()[0], train_ratio=0.1, test_ratio=0.8)
                    #     result = LREvaluator_Link()(x, y, split)
                    #     micro_f1, macro_f1 = result["micro_f1"], result["macro_f1"]
                    #
                    #
                    #
                    #
                    # if test_accu > best_test and not torch.isnan(test_accu):
                    #     best_test = test_accu
                    #     best_val = val_accu
                    #     print(
                    #         "Epoch {:05d} | CL Loss {:.4f} Val_Acc {:.4f} Test_Acc {:.4f} 🐱🐱 Best_Acc {:.4f} 🐱🐱  micro_f1{:.4f} macro_f1{:.4f}".format(
                    #             epoch, loss.item(), val_accu, test_accu, test_accu, micro_f1, macro_f1))
                    # else:
                    #     print("Epoch {:05d} | CL Loss {:.4f} Val_Acc {:.4f} Test_Acc {:.4f}".format(epoch,
                    #                                                                                 loss.item(),
                    #                                                                                 val_accu,
                    #                                                                                 test_accu))
        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
        if args.downstream_task == 'classification':
            best_test = 0
            train_trais_loss = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                hypergraph_learner.train()
                loss, hypergraph,z = loss_gcl(model, hypergraph_learner, features, anchor_adj)
                train_trais_loss.append(loss.item())
                encoder_optimizer.zero_grad()
                hyperlearner_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                hyperlearner_optimizer.step()
                if epoch % args.eval_freq == 0:
                    if args.downstream_task == 'classification':
                        model.eval()
                        hypergraph_learner.eval()
                        val_accu, test_accu, _ = evaluate_adj_by_cls(anchor_adj, features, nfeats, labels,
                                                                          nclasses, train_mask, val_mask, test_mask, args)
                        if test_accu > best_test and not torch.isnan(test_accu):
                            best_test = test_accu
                            best_val = val_accu
                            print("Epoch {:05d} | CL Loss {:.4f} Val_Acc {:.4f} Test_Acc {:.4f} 🐱🐱 Best_Acc {:.4f} 🐱🐱".format(epoch, loss.item(), val_accu, test_accu, test_accu))
                        else:
                            print("Epoch {:05d} | CL Loss {:.4f} Val_Acc {:.4f} Test_Acc {:.4f}".format(epoch, loss.item(), val_accu, test_accu))
                    elif args.downstream_task == 'clustering':
                        model.eval()
                        hypergraph_learner.eval()
                        _, embedding = model(features, hypergraph)
                        embedding = embedding.cpu().detach().numpy()
                        acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                        for clu_trial in range(args.ntrials):
                            kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(embedding)
                            predict_labels = kmeans.predict(embedding)
                            cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                            acc_mr.append(acc_)
                            nmi_mr.append(nmi_)
                            f1_mr.append(f1_)
                            ari_mr.append(ari_)
                        acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)
            Train_loss.extend(train_trais_loss)
        if args.downstream_task == 'classification':
            validation_accuracies.append(best_val.item())
            test_accuracies.append(best_test.item())
            print("Trial: ", trial + 1)
            print("Best val ACC: ", best_val.item())
            print("Best test ACC: ", best_test.item())
        elif args.downstream_task == 'clustering':
                print("Final ACC: ", acc)
                print("Final NMI: ", nmi)
                print("Final F-score: ", f1)
                print("Final ARI: ", ari)
    if args.downstream_task == 'classification' and trial != 0:
        print_results(validation_accuracies, test_accuracies)
    return validation_accuracies, test_accuracies, Train_loss

if __name__ == '__main__':
    args = config.parse()
    # datasets = ['CocitationCora', 'CoauthorshipCora', 'News20', 'CocitationCiteseer', 'Breast', 'Diabetes']
    datasets = ['CocitationCora']
    taus = [0.9]
    result_list = []
    current_iteration = 0
    for dataset in datasets:
        for tau in taus:
            args.dataset = dataset
            args.tau = tau
            validation_accuracies, test_accuracies, Train_loss = main(args)
            with open('shiyan/val_loss_{}_{}.txt'.format(tau, args.dataset), 'w') as f:
                for item in Train_loss:
                    f.write("%s\n" % item)
            f.close()