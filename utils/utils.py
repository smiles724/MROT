import torch
import os
import logging
import argparse
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build and train FullTransformer3D.")
    # set data and method
    parser.add_argument('--data', type=str, default='qm8', choices=['qm7', 'qm8', 'qm9', 'esol', 'lipo', 'freesolv'],
                        help='The molecular dataset to be used.')
    parser.add_argument('--method', type=str, default='erm', choices=['erm', 'dann', 'cdan', 'mldg', 'jdot', 'mrot'],
                        help='The approach to solve DA problems.')
    parser.add_argument('--model', type=str, default='molformer', choices=['molformer', 'egnn'])
    parser.add_argument('--meta_val_beta', type=float, default=0.1, help='The strength of the meta val loss.')
    parser.add_argument('-i', '--qm9_index', type=int, default=0, choices=[0, 1, 2, 3, 4], help='The index of selected property in QM9')
    parser.add_argument('--ratio', type=float, default=0.25, choices=[0.25, 0.5], help='The ratio of seen labels in semi-supervised learning.')

    # set hyper-paramters of MROT
    parser.add_argument('--n_clusters', type=str, default='128', help='Number of the clusters of different grains.')
    parser.add_argument('--ot_weight', type=float, default=1e10, help='Weight for OT loss.')
    parser.add_argument('--ot', type=bool, default=False, help='Whether to use the new OT distance.')
    parser.add_argument('--triplet_weight', type=float, default=1, help='Weight for triplet loss.')
    parser.add_argument('--reg1', type=float, default=1e2, help='Entropic Regularization term for OT.')
    parser.add_argument('--reg2', type=float, default=1e-1, help='Second Regularization term for OT.')

    # set the hyper-parameters of Molformer
    parser.add_argument('--atom_class', type=int, default=100, help='The default number of atom classes + 1.')
    parser.add_argument('--n_encoder', type=int, default=2, help='Number of stacked encoder.')
    parser.add_argument('--embed_dim', type=int, default=512, help='Dimension of PE, embed_dim % head == 0.')
    parser.add_argument('--head', type=int, default=4, help='Number of heads in multi-head attention.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length for the positional embedding layer.')

    # set training details
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--ep', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')

    # set training environment
    parser.add_argument('--gpu', type=str, default='0', help='Index for GPU')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save the model.')
    parser.add_argument('--model_path', default='', help='Path to load the model for visualization.')
    parser.add_argument('--save_path', default='../save/', help='Path to save the model and the logger.')

    return parser.parse_args()


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)

    # keep the cudnn stable，https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                       'error': logging.ERROR, 'crit': logging.CRITICAL}  # 日志级别关系映射

    def __init__(self, path, filename, level='info'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = logging.getLogger(path + filename)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

        th = logging.FileHandler(path + filename, encoding='utf-8')
        self.logger.addHandler(th)


if __name__ == '__main__':
    print()
