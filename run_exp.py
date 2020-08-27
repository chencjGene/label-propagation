import numpy as np

from application.views.utils.config_utils import config
from scripts.Propa.exp_config import ExpConfig
from scripts.Propa.SVHN import SVHNExp
from scripts.Propa.Cifar import CifarExp
from scripts.Propa.Cancer import CancerExp
from scripts.Propa.STL import STLExp
from scripts.Propa.OCT import OCTExp

def main(dataname, single=True, modify=False):
    if dataname == config.svhn:
        exp = SVHNExp()
    elif dataname == config.cifar10:
        exp = CifarExp()
    elif dataname == config.cancer:
        exp = CancerExp()
    elif dataname == config.stl:
        exp = STLExp()
    elif dataname == config.oct:
        exp = OCTExp()
    else:
        raise ValueError("unsupported dataname: {}".format(dataname))

    conf = ExpConfig.conf[dataname]

    if single:
       exp.run_single_exp(conf["k"], modify=modify)
    else:
        exp.run_double_exp(conf["k"])


if __name__ == "__main__":
    # main(config.stl, single=True, modify=True)
    # main(config.stl, single=True, modify=False)
    main(config.cifar10, single=False)