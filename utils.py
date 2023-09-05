import numpy as np
import scipy
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse import csc_matrix, lil_matrix
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, pairwise
from warnings import filterwarnings

filterwarnings('ignore')


# 从mat文件中读取属性、邻接矩阵和类别
def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, sp.lil_matrix):
        x = lil_matrix(x)
    return a, x, y


# 读取并预处理数据
def load_data(args):
    A_s, X_s, Y_s = load_network(args.source)
    A_t, X_t, Y_t = load_network(args.target)
    # 生成混跳图神经网络所需的邻接矩阵等输入数据
    # A_s, A_t = A_s + sp.eye(A_s.shape[0]), A_t + sp.eye(A_t.shape[0])
    X_s, X_t = X_s.todense(), X_t.todense()
    A_s_idx, A_t_idx = A_s.T.tocoo(), A_t.T.tocoo()
    A_s_idx, A_t_idx = np.array([A_s_idx.row, A_s_idx.col]), np.array([A_t_idx.row, A_t_idx.col])  # 获取非零元素坐标
    a_s = sp.coo_matrix((np.ones(A_s_idx.shape[1]), (A_s_idx[0, :], A_s_idx[1, :])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    a_t = sp.coo_matrix((np.ones(A_t_idx.shape[1]), (A_t_idx[0, :] + A_s.shape[0], A_t_idx[1, :] + A_s.shape[0])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    adj = a_s + a_t
    # 将邻接矩阵对角化
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 构建二阶邻接矩阵，该矩阵中仅包含自连接和二阶邻居连接
    adj_s = adj.dot(adj).sign()
    # 构建连边向量用于输入torch_geometric中进行学习
    edge_o = adj_o.nonzero()
    edge_idx_o = torch.tensor(np.vstack((edge_o[0], edge_o[1])), dtype=torch.long)
    edge_s = adj_s.nonzero()
    edge_idx_s = torch.tensor(np.vstack((edge_s[0], edge_s[1])), dtype=torch.long)
    features_o = np.vstack((X_s, X_t))
    labels = np.vstack((Y_s, Y_t))
    # labels = torch.FloatTensor(labels)  # 多标签分类标签
    labels = torch.argmax(torch.LongTensor(labels), dim=1)
    print('Total number of labels: {}, Total number of items: {}'.format(torch.sum(labels), labels.shape[0]))
    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x=x_o, edge_index=edge_idx_o)
    data_s = Data(edge_index=edge_idx_s)

    # 构建训练和测试样本
    source_idx = np.arange(len(Y_s))
    np.random.shuffle(source_idx)
    train_idx = source_idx[:int(len(source_idx) * args.train_ratio)]
    test_idx = np.arange(len(labels))[len(Y_s):]
    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)

    return data_o, data_s, train_idx, test_idx, Y_s, Y_t


# 节点对比损失函数
def loss_func(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# 评估多标签分类效果
def f1_scores(y_pred, y_true):
    def predict(y_t, y_p):
        top_k_list = np.array(np.sum(y_t, 1), np.int32)
        prediction = []
        for i in range(y_t.shape[0]):
            pred = np.zeros(y_t.shape[1])
            pred[np.argsort(y_p[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ['micro', 'macro']
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results['micro'], results['macro']


def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A

    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return T_S


# 读取并预处理数据
def load_gdc_data(args, alpha, eps):
    A_s, X_s, Y_s = load_network(args.source)
    A_t, X_t, Y_t = load_network(args.target)
    # 生成图扩散卷积所需的邻接矩阵等输入数据
    A_s_csr, A_t_csr = A_s.tocsr(), A_t.tocsr()
    T_s, T_t = gdc(A_s_csr, alpha, eps), gdc(A_t_csr, alpha, eps)
    adj_s, adj_t = np.nonzero(A_s), np.nonzero(A_t)
    adj_s = sp.coo_matrix((np.ones(len(adj_s[0])), (adj_s[0], adj_s[1])),
                          shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    adj_t = sp.coo_matrix((np.ones(len(adj_t[0])), (adj_t[0] + A_s.shape[0], adj_t[1] + A_s.shape[0])),
                          shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    adj = adj_s + adj_t
    print(np.count_nonzero(T_s))
    # 将其变成无向图
    T_s, T_t = T_s + T_s.T, T_t + T_t.T
    row_s, col_s = np.nonzero(T_s)
    row_t, col_t = np.nonzero(T_t)
    a_s = sp.coo_matrix((np.ones(len(row_s)), (row_s, col_s)),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    a_t = sp.coo_matrix((np.ones(len(row_t)), (row_t + A_s.shape[0], col_t + A_s.shape[0])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    # 将邻接矩阵对角化
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_x = a_s + a_t - adj_o
    X_s, X_t = X_s.todense(), X_t.todense()

    # 构建连边向量用于输入torch_geometric中进行学习
    edge_o = adj_o.nonzero()
    edge_idx_o = torch.tensor(np.vstack((edge_o[0], edge_o[1])), dtype=torch.long)
    features_o = np.vstack((X_s, X_t))
    labels = np.vstack((Y_s, Y_t))
    # labels = torch.FloatTensor(labels)  # 多标签分类标签
    # print('Total number of labels: {}, Total number of items: {}'.format(torch.sum(labels), labels.shape[0]))
    labels = torch.argmax(torch.LongTensor(labels), dim=1)
    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x=x_o, edge_index=edge_idx_o)
    edge_s = adj_x.nonzero()
    edge_index_s = torch.tensor(np.vstack((edge_s[0], edge_s[1])), dtype=torch.long)
    data_s = Data(edge_index=edge_index_s)

    # 构建训练和测试样本
    source_idx = np.arange(len(Y_s))
    np.random.shuffle(source_idx)
    train_idx = source_idx[:int(len(source_idx) * args.train_ratio)]
    test_idx = np.arange(len(labels))[len(Y_s):]
    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)

    return data_o, data_s, train_idx, test_idx, Y_s, Y_t


def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W


def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)  # 行归一化的邻接矩阵
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A


def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI

def load_ppmi_data(args):
    A_s, X_s, Y_s = load_network(args.source)
    A_t, X_t, Y_t = load_network(args.target)
    X_s, X_t = X_s.todense(), X_t.todense()

    # Compute PPMI
    Ak_s, Ak_t = AggTranProbMat(A_s, 3), AggTranProbMat(A_t, 3)  # 聚合三阶邻居重要性得到邻接矩阵
    PPMI_s, PPMI_t = ComputePPMI(Ak_s), ComputePPMI(Ak_t)  # 计算考虑三阶邻居的PPMI矩阵
    P_s, P_t = lil_matrix(MyScaleSimMat(PPMI_s)), lil_matrix(MyScaleSimMat(PPMI_t))  # 行归一化的PPMI邻接矩阵

    A_s_idx, A_t_idx = A_s.T.tocoo(), A_t.T.tocoo()
    A_s_idx, A_t_idx = np.array([A_s_idx.row, A_s_idx.col]), np.array([A_t_idx.row, A_t_idx.col])  # 获取非零元素坐标
    a_s = sp.coo_matrix((np.ones(A_s_idx.shape[1]), (A_s_idx[0, :], A_s_idx[1, :])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    a_t = sp.coo_matrix((np.ones(A_t_idx.shape[1]), (A_t_idx[0, :] + A_s.shape[0], A_t_idx[1, :] + A_s.shape[0])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    adj = a_s + a_t

    # 将邻接矩阵对角化
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    row_s, col_s = np.nonzero(P_s)
    row_t, col_t = np.nonzero(P_t)
    p_s = sp.coo_matrix((np.ones(len(row_s)), (row_s, col_s)),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    p_t = sp.coo_matrix((np.ones(len(row_t)), (row_t + A_s.shape[0], col_t + A_s.shape[0])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    pdj = p_s + p_t
    pdj_s = pdj + pdj.T.multiply(pdj.T > pdj) - pdj.multiply(pdj.T > pdj)

    edge_o = adj_o.nonzero()
    edge_idx_o = torch.tensor(np.vstack((edge_o[0], edge_o[1])), dtype=torch.long)
    features_o = np.vstack((X_s, X_t))
    labels = np.vstack((Y_s, Y_t))
    # labels = torch.FloatTensor(labels)  # 多标签分类标签
    # print('Total number of labels: {}, Total number of items: {}'.format(torch.sum(labels), labels.shape[0]))
    labels = torch.argmax(torch.LongTensor(labels), dim=1)
    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x=x_o, edge_index=edge_idx_o)
    edge_s = pdj_s.nonzero()
    edge_index_s = torch.tensor(np.vstack((edge_s[0], edge_s[1])), dtype=torch.long)
    data_s = Data(edge_index=edge_index_s)

    # 构建训练和测试样本
    source_idx = np.arange(len(Y_s))
    np.random.shuffle(source_idx)
    train_idx = source_idx[:int(len(source_idx) * args.train_ratio)]
    test_idx = np.arange(len(labels))[len(Y_s):]
    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)

    return data_o, data_s, train_idx, test_idx, Y_s, Y_t


# 读取并预处理数据, 为节点特征添加度的独热编码
def load_deg_data(args):
    A_s, X_s, Y_s = load_network(args.source)
    A_t, X_t, Y_t = load_network(args.target)
    # 生成混跳图神经网络所需的邻接矩阵等输入数据
    A_s, A_t = A_s + sp.eye(A_s.shape[0]), A_t + sp.eye(A_t.shape[0])
    X_s, X_t = X_s.todense(), X_t.todense()
    A_s_idx, A_t_idx = A_s.T.tocoo(), A_t.T.tocoo()
    A_s_idx, A_t_idx = np.array([A_s_idx.row, A_s_idx.col]), np.array([A_t_idx.row, A_t_idx.col])  # 获取非零元素坐标
    a_s = sp.coo_matrix((np.ones(A_s_idx.shape[1]), (A_s_idx[0, :], A_s_idx[1, :])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    a_t = sp.coo_matrix((np.ones(A_t_idx.shape[1]), (A_t_idx[0, :] + A_s.shape[0], A_t_idx[1, :] + A_s.shape[0])),
                        shape=(A_s.shape[0] + A_t.shape[0], A_s.shape[0] + A_t.shape[0]), dtype=np.float32)
    adj = a_s + a_t
    # 将邻接矩阵对角化
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg_o = np.count_nonzero(adj_o.toarray(), axis=1)
    deg_o = torch.unsqueeze(torch.LongTensor(deg_o), 1)
    degree_o = torch.zeros(len(deg_o), torch.max(deg_o) + 1).scatter_(1, deg_o, 1)
    # 构建二阶邻接矩阵，该矩阵中仅包含自连接和二阶邻居连接
    adj_s = adj.dot(adj).sign()
    # 构建连边向量用于输入torch_geometric中进行学习
    edge_o = adj_o.nonzero()
    edge_idx_o = torch.tensor(np.vstack((edge_o[0], edge_o[1])), dtype=torch.long)
    edge_s = adj_s.nonzero()
    edge_idx_s = torch.tensor(np.vstack((edge_s[0], edge_s[1])), dtype=torch.long)
    features_o = np.vstack((X_s, X_t))
    labels = np.vstack((Y_s, Y_t))
    # labels = torch.FloatTensor(labels)  # 多标签分类标签
    labels = torch.argmax(torch.LongTensor(labels), dim=1)
    print('Total number of labels: {}, Total number of items: {}'.format(torch.sum(labels), labels.shape[0]))
    x_o = torch.tensor(features_o, dtype=torch.float)
    feature_dim = x_o.shape[1]
    data_o = Data(x=x_o, edge_index=edge_idx_o)
    data_s = Data(edge_index=edge_idx_s)

    # 构建训练和测试样本
    source_idx = np.arange(len(Y_s))
    np.random.shuffle(source_idx)
    train_idx = source_idx[:int(len(source_idx) * args.train_ratio)]
    test_idx = np.arange(len(labels))[len(Y_s):]
    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)

    return data_o, data_s, train_idx, test_idx, Y_s, Y_t, degree_o


def get_sinkhorn_distance(M, r, c, lambda_sh, depth):
    n = M.shape[0]
    r = (r / torch.norm(r, p=1)).unsqueeze(1)
    c = (c / torch.norm(c, p=1)).unsqueeze(1)
    K = torch.exp(-lambda_sh * M)
    K_T = torch.transpose(K, 0, 1)
    v = torch.ones(n, 1) / torch.FloatTensor([n])
    # v = v.cuda()

    for i in range(depth):
        kv = torch.matmul(K, v)
        u = r / kv
        ku = torch.matmul(K_T, u)
        v = c / ku

    return torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)))

