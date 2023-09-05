import copy
import numpy as np
import faiss
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# 设置一个模型的所有参数是否需要梯度反传
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# 初始化线性映射层的参数
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class EMA:
    def __init__(self, beta, epochs):
        super(EMA, self).__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


# 梯度反转层
class GradReverse(torch.autograd.Function):
    rate = 0.0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0].neg() * GradReverse.rate
        return grad_output, None


class GRL(nn.Module):
    @staticmethod
    def forward(inp):
        return GradReverse.apply(inp)


class Encoder(nn.Module):
    """基于混跳图神经网络的对比学习编码器"""
    def __init__(self, input_dim, hidden_dim, emb_dim, dropout):
        super(Encoder, self).__init__()
        self.encoder_o1 = GATConv(input_dim, hidden_dim[0])
        self.encoder_o2 = GATConv(hidden_dim[0] * 2, hidden_dim[1])
        self.encoder_s1 = GATConv(input_dim, hidden_dim[0])
        self.encoder_s2 = GATConv(hidden_dim[0] * 2, hidden_dim[1])
        self.dropout = dropout
        self.pred = nn.Linear(hidden_dim[1] * 2, emb_dim)
        self.act = nn.Sigmoid()

    def forward(self, data_o, data_s):
        x_o, adj = data_o.x, data_o.edge_index
        adj2 = data_s.edge_index

        x1_o = F.relu(self.encoder_o1(x_o, adj))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_s = F.relu(self.encoder_s1(x_o, adj2))
        x1_s = F.dropout(x1_s, self.dropout, training=self.training)

        x1_os = torch.cat((x1_o, x1_s), dim=1)

        x2_o = self.encoder_o2(x1_os, adj)
        x2_s = self.encoder_s2(x1_os, adj2)

        x2_os = torch.cat((x2_o, x2_s), dim=1)  # 拼接一二阶聚合向量之后的节点输出向量

        pred = self.pred(x2_os)  # 将一二阶拼接向量非线性映射得到节点综合表示
        return pred


class AttnEncoder(nn.Module):
    """结合注意力的混跳图神经网络编码器"""
    def __init__(self, input_dim,  hidden_dim, emb_dim, degree_dim, dropout):
        super(AttnEncoder, self).__init__()
        self.encoder_o1 = GATConv(input_dim, hidden_dim[0])
        self.encoder_o2 = GATConv(hidden_dim[0], emb_dim)
        self.encoder_s1 = GATConv(input_dim, hidden_dim[0])
        self.encoder_s2 = GATConv(hidden_dim[0], emb_dim)
        self.dropout = dropout
        self.pred = nn.Linear(emb_dim * 2 + degree_dim, 2)

    def forward(self, data_o, data_s, degree_o):
        x_o, adj = data_o.x, data_o.edge_index
        adj2 = data_s.edge_index

        x1_o = F.relu(self.encoder_o1(x_o, adj))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_s = F.relu(self.encoder_s1(x_o, adj2))
        x1_s = F.dropout(x1_s, self.dropout, training=self.training)

        x2_o = self.encoder_o2(x1_o, adj)
        x2_s = self.encoder_s2(x1_s, adj2)

        x2_os = torch.cat((x2_o, x2_s, degree_o), dim=1)
        attn = F.softmax(self.pred(x2_os), dim=1)
        pred = torch.multiply(attn[:, :1], x2_o) + torch.multiply(attn[:, 1:], x2_s)

        return pred


class GATEncoder(nn.Module):
    """基于一阶图注意力网络的编码器"""
    def __init__(self, input_dim, hidden_dim, emb_dim, dropout):
        super(GATEncoder, self).__init__()
        self.encoder1 = GATConv(input_dim, hidden_dim[0])
        self.encoder2 = GATConv(hidden_dim[0], emb_dim)
        self.dropout = dropout

    def forward(self, data_o):
        x_o, adj = data_o.x, data_o.edge_index
        x1 = F.relu(self.encoder1(x_o, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.encoder2(x1, adj)
        return x2


class GDCN(nn.Module):
    """图扩散卷积编码器"""
    def __init__(self, input_dim, hidden_dim, emb_dim, dropout):
        super(GDCN, self).__init__()
        self.encoder1 = GATConv(input_dim, hidden_dim[0])
        self.encoder2 = GATConv(hidden_dim[0], hidden_dim[1])
        self.dropout = dropout
        self.pred = nn.Linear(hidden_dim[1], emb_dim)

    def forward(self, data_o):
        x_o, adj = data_o.x, data_o.edge_index
        x1 = F.relu(self.encoder1(x_o, adj))
        x2 = F.relu(self.encoder2(x1, adj))
        x = self.pred(x2)
        return x


class NodeClassifier(nn.Module):
    def __init__(self, emb_dim, num_class, clf_type):
        super(NodeClassifier, self).__init__()
        self.clf_type = clf_type
        self.clf = nn.Linear(emb_dim, num_class)
        std = 1 / (emb_dim / 2) ** 0.5
        nn.init.trunc_normal_(self.clf.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.clf.bias, 0.1)

    def forward(self, emb):
        pred_logit = self.clf(emb)
        return pred_logit


class DomainDiscriminator(nn.Module):
    def __init__(self, emb_dim):
        super(DomainDiscriminator, self).__init__()
        self.disrc1 = nn.Linear(emb_dim, 128)
        self.disrc2 = nn.Linear(128, 128)
        self.clf = nn.Linear(128, 2)
        std = 1 / (emb_dim / 2) ** 0.5
        nn.init.trunc_normal_(self.disrc1.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.disrc1.bias, 0.1)
        nn.init.trunc_normal_(self.disrc2.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.disrc2.bias, 0.1)
        nn.init.trunc_normal_(self.clf.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.clf.bias, 0.1)

    def forward(self, grl_feature):
        grl_feature = F.relu(self.disrc1(grl_feature))
        grl_feature = F.relu(self.disrc2(grl_feature))
        d_logit = self.clf(grl_feature)
        # d_logit = F.sigmoid(d_logit)
        return d_logit


class Neighbor(nn.Module):
    def __init__(self, args):
        super(Neighbor, self).__init__()
        self.device = args.device
        self.num_centroids = args.num_centroids
        self.num_kmeans = args.num_kmeans
        self.cluster_num_iters = args.cluster_num_iters

    # 双下划线开头的方法名是为了其被其他子类中被重写
    def __get_close_ngh_in_back(self, indices, each_k_idx, cluster_labels, back_ngh_idx, k):
        # 获取背景集合中最近的邻居
        batch_labels = cluster_labels[each_k_idx][indices]  # 本次聚类各节点的类别
        top_cluster_labels = cluster_labels[each_k_idx][back_ngh_idx]  # 每个节点的K近邻所属的聚类类别
        batch_labels = batch_labels.unsqueeze(1).expand(-1, k)
        curr_close_ngh = torch.eq(batch_labels, top_cluster_labels)  # 返回对应位置是否相等的张量矩阵
        return curr_close_ngh

    def create_sparse(self, I):
        similar = I.reshape(-1).tolist()  # 按行展开成一个列表张量
        index = np.repeat(range(I.shape[0]), I.shape[1])  # 每个元素依次重复, [0, 0, 0, 1, 1, 1...]
        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)  # 2 * (N * k)
        # 生成的是根据K近邻计算得到的节点与其最相似的k个邻居之间形成的邻接矩阵
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])
        return result

    def create_sparse_revised(self, I, all_close_ngh_in_back):
        num_sample, k = I.shape[0], I.shape[1]
        index, similar = [], []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(j)
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_ngh_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_ngh_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [num_sample, num_sample])
        return result

    def forward(self, adj, student, teacher, topk):
        num_sample, dim = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0)).detach()
        similarity += torch.eye(num_sample, device=self.device) * 10

        _, I_knn = similarity.topk(k=topk, dim=1, largest=True, sorted=True)  # N * k; N * k
        tmp = torch.LongTensor(np.arange(num_sample)).unsqueeze(-1).to(self.device)  # N * 1

        knn_neighbor = self.create_sparse(I_knn)  # N * N 邻接矩阵
        locality = knn_neighbor * adj  # 向量内积, 等价于求邻接矩阵与K紧邻的交集

        pred_labels = []
        for seed in range(self.num_kmeans):
            # 这里为什么不直接在GPU上训练呢
            kmeans = faiss.Kmeans(dim, self.num_centroids, niter=self.cluster_num_iters, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)  # 返回L2距离和最近的聚类中心
            cluster_label = I_kmeans[:, 0]  # 最近聚类中心, 相当于每个节点所属类别
            pred_labels.append(cluster_label)  # 汇集多次聚类所得的节点聚类类别

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long()  # 各节点历次聚类所得类别

        all_close_ngh_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_ngh = self.__get_close_ngh_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn,
                                                              I_knn.shape[1])
                if all_close_ngh_in_back is None:
                    all_close_ngh_in_back = curr_close_ngh
                else:
                    all_close_ngh_in_back = all_close_ngh_in_back | curr_close_ngh  # 取多次同类聚类点的并集

        all_close_ngh_in_back = all_close_ngh_in_back.to(self.device)
        globality = self.create_sparse_revised(I_knn, all_close_ngh_in_back)
        pos = locality + globality
        return pos.coalesce()._indices(), I_knn.shape[1]


class AdaCL(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(AdaCL, self).__init__()
        self.student_encoder = Encoder(input_dim, hidden_dim, args.emb_dim, args.dropout)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.num_epoch)  # 这里epoch的参数还需要好好调一下
        self.neighbor = Neighbor(args)
        self.student_predictor = nn.Sequential(nn.Linear(args.emb_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid),
                                               nn.PReLU(), nn.Linear(args.pred_hid, args.emb_dim))
        self.student_predictor.apply(init_weights)
        self.topk = args.topk
        self.node_classifier = NodeClassifier(args.emb_dim, args.num_class, args.clf_type)
        self.domain_discriminator = DomainDiscriminator(args.emb_dim)
        self.grl = GRL()

    def forward(self, data_o, data_s, neighbor, edge_weight=None):
        student = self.student_encoder(data_o, data_s)
        projection = self.student_predictor(student)
        with torch.no_grad():
            teacher = self.teacher_encoder(data_o, data_s)
        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]),
                                           [data_o.x.shape[0], data_o.x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [data_o.x.shape[0], data_o.x.shape[0]])
        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)
        # Node classifier
        pred_logit = self.node_classifier(student)
        # Domain discriminator
        grl_feature = self.grl(student)
        d_logit = self.domain_discriminator(grl_feature)
        return student, teacher, projection, ind, pred_logit, d_logit


# 将混跳图神经网络的多跳信息聚合方式改为基于Attention的方式
class AttnCL(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, degree_dim):
        super(AttnCL, self).__init__()
        self.student_encoder = AttnEncoder(input_dim, hidden_dim, args.emb_dim, degree_dim, args.dropout)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.num_epoch)
        self.neighbor = Neighbor(args)
        self.student_predictor = nn.Sequential(nn.Linear(args.emb_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid),
                                               nn.PReLU(), nn.Linear(args.pred_hid, args.emb_dim))
        self.student_predictor.apply(init_weights)
        self.topk = args.topk
        self.node_classifier = NodeClassifier(args.emb_dim, args.num_class, args.clf_type)
        self.domain_discriminator = DomainDiscriminator(args.emb_dim)
        self.grl = GRL()

    def forward(self, data_o, data_s, neighbor, edge_weight, degree_o):
        student = self.student_encoder(data_o, data_s, degree_o)
        projection = self.student_predictor(student)
        with torch.no_grad():
            teacher = self.teacher_encoder(data_o, data_s, degree_o)
        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]),
                                           [data_o.x.shape[0], data_o.x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [data_o.x.shape[0], data_o.x.shape[0]])
        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)
        # Node classifier
        pred_logit = self.node_classifier(student)
        # Domain discriminator
        grl_feature = self.grl(student)
        d_logit = self.domain_discriminator(grl_feature)
        return student, teacher, projection, ind, pred_logit, d_logit


class AdaDC(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(AdaDC, self).__init__()
        self.student_encoder = GDCN(input_dim, hidden_dim, args.emb_dim, args.dropout)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.num_epoch)  # 这里epoch的参数还需要好好调一下
        self.neighbor = Neighbor(args)
        self.student_predictor = nn.Sequential(nn.Linear(args.emb_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid),
                                               nn.PReLU(), nn.Linear(args.pred_hid, args.emb_dim))
        self.student_predictor.apply(init_weights)
        self.topk = args.topk
        self.node_classifier = NodeClassifier(args.emb_dim, args.num_class, args.clf_type)
        self.domain_discriminator = DomainDiscriminator(args.emb_dim)
        self.grl = GRL()

    def forward(self, data_o, neighbor, edge_weight=None):
        student = self.student_encoder(data_o)
        projection = self.student_predictor(student)
        with torch.no_grad():
            teacher = self.teacher_encoder(data_o)
        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]),
                                           [data_o.x.shape[0], data_o.x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [data_o.x.shape[0], data_o.x.shape[0]])
        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)
        # Node classifier
        pred_logit = self.node_classifier(student)
        # Domain discriminator
        grl_feature = self.grl(student)
        d_logit = self.domain_discriminator(grl_feature)
        return student, teacher, projection, ind, pred_logit, d_logit


# 将混跳图神经网络改为传统一阶图注意力网络, 开展消融实验对比模型效果
class GATCL(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(GATCL, self).__init__()
        self.student_encoder = GATEncoder(input_dim, hidden_dim, args.emb_dim, args.dropout)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.num_epoch)
        self.neighbor = Neighbor(args)
        self.student_predictor = nn.Sequential(nn.Linear(args.emb_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid),
                                               nn.PReLU(), nn.Linear(args.pred_hid, args.emb_dim))
        self.student_predictor.apply(init_weights)
        self.topk = args.topk
        self.node_classifier = NodeClassifier(args.emb_dim, args.num_class, args.clf_type)
        self.domain_discriminator = DomainDiscriminator(args.emb_dim)
        self.grl = GRL()

    def forward(self, data_o, neighbor, edge_weight):
        student = self.student_encoder(data_o)
        projection = self.student_predictor(student)
        with torch.no_grad():
            teacher = self.teacher_encoder(data_o)
        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]),
                                           [data_o.x.shape[0], data_o.x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [data_o.x.shape[0], data_o.x.shape[0]])
        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)
        # Node classifier
        pred_logit = self.node_classifier(student)
        # Domain discriminator
        grl_feature = self.grl(student)
        d_logit = self.domain_discriminator(grl_feature)
        return student, teacher, projection, ind, pred_logit, d_logit