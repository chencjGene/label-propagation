import numpy as np
import os
import pickle
import warnings

from scipy import sparse
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import euclidean_distances, paired_distances
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import csgraph, csr_matrix
from time import time
from scipy.stats import entropy
from tqdm import tqdm

# from application.views.model_utils import Data
from application.views.model_utils.data_old import Data
from application.views.utils.config_utils import config
from application.views.utils.log_utils import logger
from application.views.utils.embedder_utils import Embedder

from scripts.exp.subset_selection import weight_selection, uncertainty_selection
from application.views.model_utils.model_helper import uncertainty_selection as new_uncertainty_selection

def build_laplacian_graph(affinity_matrix):
    instance_num = affinity_matrix.shape[0]
    laplacian = csgraph.laplacian(affinity_matrix, normed=True)
    laplacian = -laplacian
    if sparse.isspmatrix(laplacian):
        diag_mask = (laplacian.row == laplacian.col)
        laplacian.data[diag_mask] = 0.0
    else:
        laplacian.flat[::instance_num + 1] = 0.0  # set diag to 0.0
    return laplacian

def propagation(graph_matrix, affinity_matrix, train_y, alpha=0.2, max_iter=15,
                tol=1e-12, process_record=False, normalized=False, k=6, modify=False, neighbors=None):
    y = np.array(train_y)
    # label construction
    # construct a categorical distribution for classification only
    classes = np.unique(y)
    classes = (classes[classes != -1])

    modified_matrix = affinity_matrix.copy()
    graph_matrix = build_laplacian_graph(modified_matrix)

    n_samples, n_classes = len(y), len(classes)

    if (alpha is None or alpha <= 0.0 or alpha >= 1.0):
        raise ValueError('alpha=%s is invalid: it must be inside '
                         'the open interval (0, 1)' % alpha)
    y = np.asarray(y)

    # initialize distributions
    label_distributions_ = np.zeros((n_samples, n_classes))
    for label in classes:
        label_distributions_[y == label, classes == label] = 1

    y_static_labeled = np.copy(label_distributions_)
    y_static = y_static_labeled * (1 - alpha)

    if sparse.isspmatrix(graph_matrix):
        graph_matrix = graph_matrix.tocsr()

    n_iter_ = 1
    for _ in range(max_iter):
        label_distributions_a = safe_sparse_dot(
            graph_matrix, label_distributions_)

        label_distributions_ = np.multiply(
            alpha, label_distributions_a) + y_static

        n_iter_ += 1

        # calculate entropy
        label = label_distributions_.copy()
        normalizer = np.sum(label, axis=1)[:, np.newaxis]
        normalizer = normalizer + 1e-20
        label /= normalizer
        ent = entropy(label.T + 1e-20)
        print("entropy: ", ent.sum())
        unpropagated_num = sum(ent > (np.log(label.shape[1]) - 0.001))
        print("iter:{}\tunpropagated_num: {}"\
            .format(n_iter_, unpropagated_num))

        # calculate entropy
        label = label_distributions_.copy()
        normalizer = np.sum(label, axis=1)[:, np.newaxis]
        normalizer = normalizer + 1e-20
        label /= normalizer
        ent = entropy(label.T + 1e-20)
        # print("entropy: ", ent.sum())

        # modify graph
        # if modify and n_iter_ < 9:
        if modify:
            # modified_matrix = uncertainty_selection(ent, label, modified_matrix,
            #                                             label_distributions_, alpha, y_static, train_y,
            #                                             build_laplacian_graph, affinity_matrix, neighbors)
            modified_matrix = weight_selection(graph_matrix.tocsr(), affinity_matrix, label_distributions_, alpha, y_static, ent, label)

        graph_matrix = build_laplacian_graph(modified_matrix)
    else:
        warnings.warn(
            'max_iter=%d was reached without convergence.' % max_iter,
            category=ConvergenceWarning
        )
    # normalization
    normalizer = np.sum(label_distributions_, axis=1)[:, np.newaxis]
    normalizer = normalizer + 1e-20
    label_distributions_ /= normalizer


    return label_distributions_