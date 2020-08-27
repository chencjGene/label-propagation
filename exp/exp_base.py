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
from application.views.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data

from scripts.exp.subset_selection import weight_selection, uncertainty_selection
from scripts.Propa.propagation_helper import propagation, build_laplacian_graph
from application.views.model_utils.model_helper import new_propagation

def csr_to_impact_matrix(neighbor_result, instance_num, max_neighbors):
    neighbors = np.zeros((instance_num, max_neighbors)).astype(int)
    neighbors_weight = np.zeros((instance_num, max_neighbors))
    for i in range(instance_num):
        start = neighbor_result.indptr[i]
        end = neighbor_result.indptr[i + 1]
        j_in_this_row = neighbor_result.indices[start:end]
        data_in_this_row = neighbor_result.data[start:end]
        sorted_idx = data_in_this_row.argsort()
        assert (len(sorted_idx) == max_neighbors)
        j_in_this_row = j_in_this_row[sorted_idx]
        data_in_this_row = data_in_this_row[sorted_idx]
        neighbors[i, :] = j_in_this_row
        neighbors_weight[i, :] = data_in_this_row
    return neighbors, neighbors_weight

class ExpBase(object):
    def __init__(self, dataname):
        """
        this the parent class for run experiments
        :param dataname: dataname from config
        """
        self.dataname = dataname
        self.data_dir = os.path.join(config.data_root, dataname)
        # self.raw_data_dir = os.path.join(config.raw_data_root, dataname)
        self.selected_dir = None
        check_dir(self.data_dir)

        self._processing()

    def _processing(self):
        data = Data(self.dataname)
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


    def preprocess_neighbors(self, rebuild=False, save=True):
        neighbors_model_path = os.path.join(self.selected_dir, "neighbors_model" + ".pkl")
        neighbors_path = os.path.join(self.selected_dir, "neighbors" + ".npy")
        neighbors_weight_path = os.path.join(self.selected_dir,
                                             "neighbors_weight" + ".npy")
        test_neighbors_path = os.path.join(self.selected_dir, "test_neighbors" + ".npy")
        test_neighbors_weight_path = os.path.join(self.selected_dir, "test_neighbors_weight" + ".npy")
        if os.path.exists(neighbors_model_path) and \
                os.path.exists(neighbors_path) and \
                os.path.exists(test_neighbors_path) and rebuild == False:
            print("neighbors and neighbor_weight exist!!!")
            neighbors = np.load(neighbors_path)
            neighbors_weight = np.load(neighbors_weight_path)
            test_neighbors = np.load(test_neighbors_path)
            self.test_neighbors = test_neighbors
            return neighbors, neighbors_weight, test_neighbors
        print("neighbors and neighbor_weight  do not exist, preprocessing!")
        train_num = self.train_X.shape[0]
        train_y = np.array(self.train_y)
        test_num = self.test_X.shape[0]
        max_neighbors = min(len(train_y), 200)
        print("data shape: {}, labeled_num: {}"
                    .format(str(self.train_X.shape), sum(train_y != -1)))
        nn_fit = NearestNeighbors(7, n_jobs=-4).fit(self.train_X)
        print("nn construction finished!")
        neighbor_result = nn_fit.kneighbors_graph(nn_fit._fit_X,
                                                  max_neighbors,
                                                  # 2,
                                                  mode="distance")
        test_neighbors_result = nn_fit.kneighbors_graph(self.test_X,
                                                        max_neighbors,
                                                        mode="distance")
        print("neighbor_result got!")
        neighbors, neighbors_weight = csr_to_impact_matrix(neighbor_result,
                                                                     train_num, max_neighbors)
        test_neighbors, test_neighbors_weight = csr_to_impact_matrix(test_neighbors_result,
                                                                               test_num, max_neighbors)
        self.test_neighbors = test_neighbors

        print("preprocessed neighbors got!")

        # save neighbors information
        if save:
            pickle_save_data(neighbors_model_path, nn_fit)
            np.save(neighbors_path, neighbors)
            np.save(neighbors_weight_path, neighbors_weight)
            np.save(test_neighbors_path, test_neighbors)
            np.save(test_neighbors_weight_path, test_neighbors_weight)
        return neighbors, neighbors_weight, test_neighbors

    def construct_graph(self, n_neighbor=None, weight=False):
        # create neighbors buffer
        neighbors, neighbors_weight, test_neighbors = self.preprocess_neighbors()
        self.neighbors = neighbors

        # # load neighbors information
        instance_num = neighbors.shape[0]
        print("train_y", self.train_y.shape)

        # get knn graph in a csr form
        indptr = [i * n_neighbor for i in range(instance_num + 1)]
        print("get indptr")
        indices = neighbors[:, :n_neighbor].reshape(-1).tolist()
        print("get indices")
        if not weight:
            data = neighbors[:, :n_neighbor].reshape(-1)
            print("get data")
            data = (data * 0 + 1.0).tolist()
        else:
            data = neighbors_weight[:, :n_neighbor].reshape(-1).tolist()
        print("get data in connectivity")
        affinity_matrix = sparse.csr_matrix((data, indices, indptr),
                                            shape=(instance_num, instance_num))
        affinity_matrix = affinity_matrix + affinity_matrix.T
        affinity_matrix = sparse.csr_matrix((np.ones(len(affinity_matrix.data)).tolist(),
                                                affinity_matrix.indices, affinity_matrix.indptr),
                                            shape=(instance_num, instance_num))

        print("affinity_matrix construction finished!!")

        return affinity_matrix

    def adaptive_evaluation_unasync(self, affinity_matrix, test_X, test_y, test_neighbors, pred):
        affinity_matrix.setdiag(0)
        print("neighbor_result got!")
        estimate_k = 3
        s = 0
        labels = []
        adaptive_ks = []
        for i in tqdm(range(test_X.shape[0])):
            j_in_this_row = test_neighbors[i, :]
            j_in_this_row = j_in_this_row[j_in_this_row != -1]
            estimated_idxs = j_in_this_row[:estimate_k]
            # estimated_idxs = [m[i] for i in estimated_idxs]
            adaptive_k = affinity_matrix[estimated_idxs, :].sum() / estimate_k
            selected_idxs = j_in_this_row[:int(adaptive_k)]
            # selected_idxs = [m[i] for i in selected_idxs]
            p = pred[selected_idxs].sum(axis=0)
            labels.append(p.argmax())
            s += adaptive_k
            adaptive_ks.append(adaptive_k)

        acc = accuracy_score(test_y, labels)
        confusion_mat = confusion_matrix(test_y, labels)
        print("exp accuracy: {}".format(acc))
        return labels, np.array(adaptive_ks), acc

    def run_single_exp(self, n_neighbor=4, modify=False):
        affinity_matrix = self.construct_graph(n_neighbor=n_neighbor)
        laplacian = build_laplacian_graph(affinity_matrix)
        pred_dist = propagation(laplacian, affinity_matrix, self.train_y,
                        alpha=0.2, process_record=True,
                        normalized=False, k=n_neighbor, modify=modify, neighbors=self.neighbors)
        # _, pred_dist = new_propagation(affinity_matrix, self.train_y, 0.2)
        pred_y = pred_dist.argmax(axis=1)
        acc = accuracy_score(self.train_gt, pred_y)
        print("accuracy:", acc)
        labels, _, acc = self.adaptive_evaluation_unasync(affinity_matrix,
                                         self.test_X,
                                         self.test_gt,
                                         self.test_neighbors,
                                         pred_dist)
        pre = "old"
        if modify:
            pre = "new"
        np.save(os.path.join(self.data_dir, pre+ "_labels.npy"), labels)
        return acc



    def run_double_exp(self, n_neighbor=4):
        affinity_matrix = self.construct_graph(n_neighbor=n_neighbor)

        # no modified version
        old_acc = self.run_single_exp(n_neighbor, modify=False)

        # modified version
        new_acc = self.run_single_exp(n_neighbor, modify=True)
        return old_acc, new_acc
