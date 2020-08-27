import numpy as np
import os

from application.views.utils.config_utils import config
from scripts.Propa.exp_base import ExpBase


class CifarExp(ExpBase):
    def __init__(self):
        dataname = config.cifar10
        super(CifarExp, self).__init__(dataname)

