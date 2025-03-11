import os

import pandas as pd
import torch
import torch.nn as nn
from dhg import Hypergraph
from sklearn import metrics, datasets
from munkres import Munkres
import numpy as np
from tqdm import tqdm

from sample import start_sample
from sample_config import config


def toHypergraph(H):
    HyperList = [torch.where(H[:, i] != 0)[0].tolist() for i in range(H.shape[1])]
    HG = Hypergraph(H.shape[0], HyperList)
    return HG
def print_results(validation_accu, test_accu):
    s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
    s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu), np.std(test_accu))
    print(s_val)
    print(s_test)
class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label
    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, print_results=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        if print_results:
            print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
                  .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
                  'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
                  .format(precision_micro, recall_micro, nmi, adjscore))

        return acc, nmi, f1_macro, adjscore
def read_diabetes():
    df = pd.read_csv('diabetes.csv')
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X = torch.tensor(X.values, dtype=torch.float32)
    Y = torch.tensor(Y.values, dtype=torch.long)
    return X, Y
def read_breast_cancer():
    cancer_data = datasets.load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target
    X = torch.tensor(X, dtype=torch.float32)
    lbl = torch.tensor(y, dtype=torch.long)
    return X, lbl

def neg_sampler(E, dataset_name):
    print('============>>>>>>>>>开始进行负采样<<<<<<<<<============')
    if not os.path.exists('{}/hyperlink.txt'.format(dataset_name)):
        hypergraph_matrix = E.H.to_dense()
        nonzero_positions = {i: [idx for idx, val in enumerate(column) if val != 0] for i, column in
                             enumerate(zip(*hypergraph_matrix), 1)}
        with open('{}/hyperlink.txt'.format(dataset_name), 'w') as file:
            for col, positions in nonzero_positions.items():
                positions_str = ', '.join(map(str, positions))
                file.write(f"{positions_str}\n")
    args = config().parse_args()
    args.hyperedges = '{}/hyperlink.txt'.format(dataset_name)
    args.output = '{}/neg_hyperlink.txt'.format(dataset_name)
    if not os.path.exists('{}/neg_hyperlink.txt'.format(dataset_name)):
        start_sample(args)
    if os.path.exists('{}/neg_hyperlink.txt'.format(dataset_name)):
        with open('{}/neg_hyperlink.txt'.format(dataset_name), 'r') as file:
            lines = file.readlines()
        neg_lists = []
        for line in tqdm(lines):
            elements = line.strip().split(', ')
            int_list = [int(elem) for elem in elements]
            neg_lists.append(int_list)
    return neg_lists