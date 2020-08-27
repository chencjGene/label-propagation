import numpy as np
import os

from application.views.utils.config_utils import config
from scripts.Propa.exp_base import ExpBase


class SVHNExp(ExpBase):
    def __init__(self):
        dataname = config.svhn
        super(SVHNExp, self).__init__(dataname)

