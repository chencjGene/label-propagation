import numpy as np
import os

from application.views.utils.config_utils import config
from application.views.model_utils.data_old import Data
from scripts.Propa.exp_base import ExpBase


class OCTExp(ExpBase):
    def __init__(self):
        dataname = config.oct
        super(OCTExp, self).__init__(dataname)

    def _processing(self):
        data = Data(self.dataname, labeled_num=1000, total_num=10000)
        self.selected_dir = data.selected_dir
        self.train_X = data.get_train_X()
        self.train_X = np.squeeze(self.train_X)
        self.train_y = data.get_train_label()
        self.train_y = np.squeeze(self.train_y)
        self.train_gt = data.get_train_ground_truth()
        self.train_gt = np.squeeze(self.train_gt)
        self.test_X = data.get_test_X()
        self.test_X = np.squeeze(self.test_X)
        self.test_gt = data.get_test_ground_truth()
        self.test_gt = np.squeeze(self.test_gt)
        self.train_gt = np.array(self.train_gt)
        print("train_X shape:", self.train_X.shape)
        print("test_X shape:", self.test_X.shape)