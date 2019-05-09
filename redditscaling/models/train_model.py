import argparse
import os.path as osp

import starwrap as sw

from redditscaling.config import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train StarSpace model')

    parser.add_argument('trainFile', type=str)
    parser.add_argument('validationFile', type=str)
    parser.add_argument('modelName', type=str)

    parser.add_argument('--initRandSd', type=float, default=0.01)
    parser.add_argument('--adagrad', action='store_true')
    parser.add_argument('--ngrams', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--thread', type=int, default=16)
    parser.add_argument('--dim', type=int, default=200)
    parser.add_argument('--trainMode', type=int, default=0)
    parser.add_argument('--label', type=str, default='__label__')
    parser.add_argument('--similarity', type=str, default='cosine')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--normalizeText', action='store_true')
    parser.add_argument('--minCount', type=int, default=5)
    parser.add_argument('--validationPatience', type=int, default=10)

    args = parser.parse_args()
    args_dict = vars(args)
    arg = sw.args()
    for key, value in args_dict.items():
        if key != 'modelName':
            setattr(arg, key, value)

    sp = sw.starSpace(arg)
    sp.init()
    sp.train()

    sp.saveModel(osp.join(config.model_dir, args.modelName))
    sp.saveModelTsv(osp.join(config.model_dir, f'{args.modelName}.tsv'))
