import os.path as osp

import torch

CURRENT_PATH = osp.dirname(osp.realpath(__file__))


class Config:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = osp.realpath(osp.join(CURRENT_PATH, '..', 'data'))
        self.model_dir = osp.realpath(osp.join(CURRENT_PATH, '..', 'models'))


config = Config()
