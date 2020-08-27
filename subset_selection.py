import numpy as np
from tqdm import tqdm
import math

from sklearn.utils.extmath import safe_sparse_dot
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
from scipy.stats import entropy
from scipy.sparse import csr_matrix, coo_matrix
from time import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

from numba import jit, float64, int32


#@autojit
def _sparse_mult4(a, b, cd, cr, cc):
    N = cd.size
    data = np.empty_like(cd)
    for i in range(N):
        num = 0.0
        for j in range(a.shape[1]):
            num += a[cr[i], j] * b[j, cc[i]]
        data[i] = cd[i]*num
    return data


_fast_sparse_mult4 = \
    jit(float64[:,:](float64[:,:],float64[:,:],float64[:],int32[:],int32[:]))(_sparse_mult4)


def sparse_numba(a,b,c):
    """Multiply sparse matrix `c` by np.dot(a,b) using Numba's jit."""
    assert c.shape == (a.shape[0],b.shape[1])
    data = _fast_sparse_mult4(a,b,c.data,c.row,c.col)
    return coo_matrix((data,(c.row,c.col)),shape=(a.shape[0],b.shape[1]))


def multi_processing_cost(W, graph_matrix, label_distributions_, gt, regularization_weight, alpha, y_static):
        tmp = graph_matrix.copy()
        W[W <= 0] = 1e-20
        tmp.data = tmp.data * W
        # W must be non-zero
        P = safe_sparse_dot(tmp, label_distributions_)
        P = np.multiply(alpha, P) + y_static
        cost_1 = entropy(P.T+1e-20).sum()
        cost_2 = regularization_weight * ((P-gt)**2).sum()
        cost = cost_1 + cost_2
        # print("total_cost: {},\tcost_1: {},\tcost_2: {}".format(cost, cost_1, cost_2))
        # cost = entropy(P.T+1e-20).sum() # for DEBUG
        # cost = 0.1 * ((P-gt)**2).sum() # for DEBUG
        # cost = P.sum() # for DEBUG
        # print("cost", cost)
        return cost


def sub_task(W0, graph_matrix, label_distributions_, origin_cost, gt, start, end):
    print("start sub task: {}-{}".format(start, end))
    WG = np.zeros(end-start)
    for idx, i in enumerate(range(start, end)):
        delta = 0.000001
        tmp_W = W0.copy()
        tmp_W[i] = tmp_W[i] + delta
        cost = multi_processing_cost(tmp_W, graph_matrix, label_distributions_, gt)
        WG[idx] = (cost - origin_cost) / delta
    print("end sub task: {}-{}".format(start, end))
    return WG.tolist()

def weight_selection(graph_matrix, origin_graph, label_distributions_, alpha, y_static, ent, label):
    gt = safe_sparse_dot(graph_matrix, label_distributions_)
    gt = np.multiply(alpha, gt) + y_static

    # regularization_weight = 1e11 * 0.5
    regularization_weight = 1e8

    ind = ent > np.log(label.shape[1]) * 0.15
    ind[ent > (np.log(label.shape[1]) - 0.001)] = False

    graph_matrix.eliminate_zeros()
    # graph_matrix = graph_matrix.tocsr()
    graph_matrix[:, ind] = graph_matrix[:, ind] * 0

    W0 = graph_matrix.data > 0
    W0 = np.array(W0).astype(float)
    bounds = [(0,1) for i in range(len(W0))]

    def cost_function(W):
        return multi_processing_cost(W, graph_matrix, label_distributions_, gt, regularization_weight, alpha, y_static)
        # tmp = graph_matrix.copy()
        # tmp.data = tmp.data * W
        # P = safe_sparse_dot(tmp, label_distributions_)
        # # cost = entropy(P.T+1e-20).sum() + 0.1 * ((P-gt)**2).sum()
        # # cost = entropy(P.T+1e-20).sum() # for DEBUG
        # cost = 0.1 * ((P-gt)**2).sum() # for DEBUG
        # # print("cost", cost)
        # return cost

    def cost_der(W):
        t0 = time()
        coo = graph_matrix.tocoo()
        tmp = graph_matrix.copy()
        tmp.data = tmp.data * W
        P = safe_sparse_dot(tmp, label_distributions_)
        P = np.multiply(alpha, P) + y_static + 1e-20
        normalizer = np.sum(P, axis=1)[:, np.newaxis]
        normalizer = normalizer + 1e-20
        norm_P = P / normalizer + 1e-20
        log_P = np.log(norm_P)
        P_sum = (P.sum(axis=1) + 1e-20)
        G11 = (norm_P * log_P).sum(axis=1) / P_sum

        G11 = G11[np.newaxis, :].repeat(axis=0, repeats=label_distributions_.shape[1])
        G11 = G11.T
        # G1 = np.dot(G11, label_distributions_.T)
        # G1 = np.dot(G11[:, np.newaxis], G12[np.newaxis, :])
        # G1 = sparse_numba(G11[:, np.newaxis], G12[np.newaxis, :], coo)

        # G2 = np.dot(log_P / P_sum[:, np.newaxis], label_distributions_.T)
        # G3 = np.dot(P-gt, label_distributions_.T)
        # G = G1 - G2 + 0.1 * G3
        # G = np.dot(G11 - log_P / P_sum[:, np.newaxis] + 0.1 * (P-gt), label_distributions_.T)
        G = sparse_numba(G11 - log_P / P_sum[:, np.newaxis] + regularization_weight * (P-gt), label_distributions_.T, coo)
        # G = 0.1 * G3 # for DEBUG
        # G = G1 - G2 # for DEBUG
        # final_G = graph_matrix.data * G[graph_matrix.nonzero()[0], graph_matrix.nonzero()[1]]
        final_G = G.tocsr().data * alpha
        # print("t4:", time() - t0)

        return final_G

    def simple_cost_der(W):
        tmp = graph_matrix.copy()
        tmp.data = tmp.data * W
        P = safe_sparse_dot(tmp, label_distributions_) + 1e-20
        L = label_distributions_.sum(axis=1)
        G = L[np.newaxis,:].repeat(axis=0, repeats=len(L))
        final_G = graph_matrix.data * G[graph_matrix.nonzero()[0], graph_matrix.nonzero()[1]]
        return final_G


    # # bruce force methods for gradient
    # WG = np.zeros(W0.shape)
    # origin_cost = cost_function(W0)
    #
    # cpu_kernel = 40
    # step_size = math.ceil(len(WG) / cpu_kernel)
    # start_ends = []
    # for i in range(cpu_kernel):
    #     start_ends.append([i*step_size,
    #                        min((i+1)*step_size, len(WG))])
    # multi_p_res = [None for i in range(cpu_kernel)]
    # pool = Pool()
    # res = [pool.apply_async(sub_task,
    #         (W0, graph_matrix, label_distributions_, origin_cost, gt, start, end))
    #             for start, end in start_ends]
    # for idx, r in enumerate(res):
    #     multi_p_res[idx] = r.get()
    # WG = []
    # for l in multi_p_res:
    #     WG.extend(l)
    # WG = np.array(WG)

    # WG = np.load("WG.npy")
    #
    # G = cost_der(W0)

    # plt.scatter(WG, G)
    # plt.show()

    res = minimize(cost_function, W0, method="L-BFGS-B", jac=cost_der, bounds=bounds,
                   options={'disp': False}, tol=1e-3).x
    # res = gradient_descent(cost_function, cost_der, W0, learning_rate=0.01)
    # print(res)

    W = res
    graph_matrix.data = (W > 0.5).astype(int)

    # postprocess for case where some instances are not propagated to
    num_point_to = graph_matrix.sum(axis=1).reshape(-1)
    num_point_to = np.array(num_point_to).reshape(-1)
    ids = np.array(range(len(num_point_to)))[num_point_to==0]
    for id in ids:
        point_to_idxs = origin_graph[:,id].nonzero()[0]
        dists = label_distributions_[point_to_idxs, :]
        labels = dists.argmax(axis=1)
        bins = np.bincount(labels)
        max_labels = bins.argmax()
        point_to_idxs = point_to_idxs[labels==max_labels]
        for p in point_to_idxs:
            graph_matrix[id, p] = 1
        a = 1

    removed_num = len(W0) - graph_matrix.data.sum()
    # print("removed_num:", removed_num)
    return graph_matrix


    # exit()

def _find_unconnected_nodes(affinity_matrix, labeled_id):
    # logger.info("Finding unconnected nodes...")
    edge_indices = affinity_matrix.indices
    edge_indptr = affinity_matrix.indptr
    node_num = edge_indptr.shape[0] - 1
    connected_nodes = np.zeros((node_num))
    connected_nodes[labeled_id] = 1

    iter_cnt = 0
    while True:
        new_connected_nodes = affinity_matrix.dot(connected_nodes)+connected_nodes
        new_connected_nodes = new_connected_nodes.clip(0, 1)
        iter_cnt += 1
        if np.allclose(new_connected_nodes, connected_nodes):
            break
        connected_nodes = new_connected_nodes
    unconnected_nodes = np.where(new_connected_nodes<1)[0]
    # logger.info("Find unconnected nodes end. Count:{}, Iter:{}".format(unconnected_nodes.shape[0], iter_cnt))
    return unconnected_nodes

def correct_unconnected_nodes(affinity_matrix, train_y, neighbors):
    print("begin correct unconnected nodes...")
    np.random.seed(123)
    correted_nodes = []
    affinity_matrix = affinity_matrix.copy()
    labeled_ids = np.where(train_y > -1)[0]
    iter_cnt = 0
    while True:
        unconnected_ids = _find_unconnected_nodes(affinity_matrix, labeled_ids)
        if unconnected_ids.shape[0] == 0:
            print("No correcnted nodes after {} iteration. Correction finished.".format(iter_cnt))
            return affinity_matrix
        else:
            while True:
                corrected_id = np.random.choice(unconnected_ids)
                k_neighbors = neighbors[corrected_id]
                find = False
                for neighbor_id in k_neighbors:
                    if neighbor_id not in unconnected_ids:
                        find = True
                        iter_cnt += 1
                        affinity_matrix[corrected_id, neighbor_id] = 1
                        correted_nodes.append([corrected_id, neighbor_id])
                        break
                if find:
                    break

def uncertainty_selection(ent, label, modified_matrix,
                          label_distributions_, alpha, y_static, train_y,
                          build_laplacian_graph, origin_graph, neighbors):
    graph_matrix = build_laplacian_graph(modified_matrix)
    P = safe_sparse_dot(graph_matrix, label_distributions_)
    P = np.multiply(alpha, P) + y_static
    pre_ent = entropy(P.T + 1e-20)

    ind = ent > np.log(label.shape[1]) * 0.1
    ind[ent > (np.log(label.shape[1]) - 0.001)] = False
    modified_matrix[:, ind] = modified_matrix[:, ind] * 0

    graph_matrix = build_laplacian_graph(modified_matrix)
    P = safe_sparse_dot(graph_matrix, label_distributions_)
    P = np.multiply(alpha, P) + y_static
    next_ent = entropy(P.T + 1e-20)

    # postprocess for case where some instances are not propagated to
    # num_point_to = modified_matrix.sum(axis=1).reshape(-1)
    # num_point_to = np.array(num_point_to).reshape(-1)
    # ids = np.array(range(len(num_point_to)))[num_point_to == 0]
    # for id in ids:
    #     point_to_idxs = origin_graph[:, id].nonzero()[0]
    #     dists = label_distributions_[point_to_idxs, :]
    #     labels = dists.argmax(axis=1)
    #     bins = np.bincount(labels)
    #     max_labels = bins.argmax()
    #     point_to_idxs = point_to_idxs[labels == max_labels]
    #     for p in point_to_idxs:
    #         modified_matrix[id, p] = 1
    #     a = 1
    modified_matrix = correct_unconnected_nodes(modified_matrix, train_y, neighbors)

    print("removed_num: {}, ent1: {}, ent2: {}, ent_gain: {}"
            .format(ind.sum(), pre_ent.sum(), next_ent.sum(), pre_ent.sum() - next_ent.sum()))
    return modified_matrix

def gradient_descent(cost_function, cost_der, W0, learning_rate=1.0, max_iters=3000):
    W = W0
    pre_loss = 0
    for i in range(max_iters):
        grad_cur = cost_der(W)
        # if abs(grad_cur).sum() < 1e2:
        #     break
        W = W - learning_rate * grad_cur
        new_loss = cost_function(W)
        if abs(new_loss - pre_loss) < 1e-3:
            break
        pre_loss = new_loss
        # print("grad_cur:", abs(grad_cur).sum())
        print("iter_num: {}, cost: {}".format(i, pre_loss))
    return W