import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def my_load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data2/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data2/ind.{}.test.index".format(dataset_str))
    print(len(test_idx_reorder))
    test_idx_range = np.sort(test_idx_reorder)
  
    if dataset_str == 'citeseer':
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    print("Label matrix:" + str(labels.shape))

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def my_load_data_(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data2/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    x = x.toarray()
    tx = tx.toarray()
    allx = allx.toarray()

    test_idx_reorder = parse_index_file("data2/ind.{}.test.index".format(dataset_str))
    test_index = np.sort(test_idx_reorder)

    s = test_index.min()
    t = test_index.max()
    tx_zero = np.zeros(tx.shape[1], dtype=np.float).reshape(1, -1)
    ty_zero = np.zeros(ty.shape[1]).reshape(1, -1)
    ty_zero[0,5] = 1

    for i in range(s, t + 1):
        if i not in test_index:
            arr_i = np.array(i).reshape(1, )
            test_index = np.concatenate((test_index, arr_i), axis=0)
            tx = np.concatenate((tx, tx_zero), axis=0)
            ty = np.concatenate((ty, ty_zero), axis=0)

        # 通过索引index划分数据集
    train_index = np.arange(y.shape[0])  # 第0维是训练节点的个数
    val_index = np.arange(y.shape[0], y.shape[0] + 500)  # 再往后找500个是验证集
    sorted_test_index = sorted(test_index)  # 对测试索引进行从小到大排序（不改变原列表）

    # 将训练节点和测试节点特征进行拼接-->按行拼接，得到【全图】的特征表示x
    x = np.concatenate((allx, tx), axis=0)
    # 将训练节点和测试节点one-hot标签-->按行拼接+按列max，得到【全图】的（数值）标签y
    y = np.concatenate((ally, ty), axis=0)

        # x，y也改变为相应的测试顺序？？？
    x[test_index] = x[sorted_test_index]
    x = normalize_features(x)
    y[test_index] = y[sorted_test_index]
        # x的第0维是节点数量
    num_nodes = x.shape[0]
        # 初始化mask向量
    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)
    # 通过索引为mask赋值
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True
        # 邻接字典（表）
    adjacency_dict = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adjacency_dict = normalize_adj(adjacency_dict + sp.eye(adjacency_dict.shape[0]))

        # 打印数据的信息
    print("Node's feature shape: ", x.shape)
    print("Node's label shape: ", y.shape)
    print("Adjacency's shape: ", adjacency_dict.shape)
    print("Number of training nodes: ", train_mask.sum())
    print("Number of validation nodes: ", val_mask.sum())
    print("Number of test nodes: ", test_mask.sum())

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]

    adj = torch.FloatTensor(np.array(adjacency_dict.todense()))
    features = torch.FloatTensor(np.array(x))
    
    print(y.shape)
    labels = torch.LongTensor(np.where(y)[1])
    print(labels.size())    

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test