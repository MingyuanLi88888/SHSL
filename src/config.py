import argparse
def parse():
    parser = argparse.ArgumentParser(description='Hypergraph struct', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset', type=str, default='CocitationCora',
                        choices=['CocitationCora',  'CoauthorshipCora', 'CoauthorshipDBLP',
                                 'CocitationCiteseer', 'CocitationPubmed',
                                 'YelpRestaurant', 'WalmartTrips','HouseCommittees', 'News20',
                                 'Cooking200', 'diabetes', 'breast_cancer'])
    # 'YelpRestaurant', 'WalmartTrips', 'HouseCommittees', 'News20' 这四个数据集没有train mask等
    # 'Cooking200' 数据集并没有节点特征，仅有超图结构。
    parser.add_argument('-gsl_mode', type=str, default="structure_inference",
                            choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-downstream_task', type=str, default='classification',
                            choices=['classification', 'clustering', 'multi-task'])
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-ntrials', type=int, default=1)

    # Init Hypergraph
    parser.add_argument('-num_clusters', type=int, default=15)
    parser.add_argument('-fea_dim', type=int, default=10)
    parser.add_argument('-i_bias', type=int, default=6)
    parser.add_argument('-hypergraph_learner', type=str, default='att',
                                             choices=['iecs', 'att'])


    # Graph Encoder
    parser.add_argument('-epochs', type=int, default=800)
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-embedding_dim', type=int, default=256)
    parser.add_argument('-proj_dim', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-encoder_lr', type=float, default=0.001)
    parser.add_argument('-encoder_wd', type=float, default=0.00000)
    parser.add_argument('-anchor_featuremask', type=float, default=0.3)
    parser.add_argument('-hgraphlearner_featuremask', type=float, default=0.3)
    parser.add_argument('-anchor_dropedge_rate', type=float, default=0.3)
    parser.add_argument('-hgyper_dropedge_rate', type=float, default=0.3)

    parser.add_argument('-tau', type=float, default=0.7)
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('-Encoder_type', type=str, default='UniSAGE',
                        choices=['HyperGCN', 'UniSAGE', 'UniGIN', 'HGNNP', 'UniGAT'])




    # Classifier
    parser.add_argument('-hidden_dim_classifier', type=int, default=32)
    parser.add_argument('-lr_classifier', type=float, default=0.01)
    parser.add_argument('-wd_classifier', type=float, default=0.0000)
    parser.add_argument('-dropout_classifier', type=float, default=0.5)
    parser.add_argument('-epochs_classifier', type=int, default=200)












    return parser.parse_args()