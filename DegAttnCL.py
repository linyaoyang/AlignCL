"""基于节点对比学习的跨网络节点分类算法
1. 利用混跳图神经网络挖掘网络局部和全局结构信息;
2. 利用节点对比学习使节点表示聚类;
3. 在领域自适应损失中添加了熵损失, 使源网络和目标网络预测类别分布一致"""
import faiss
import numpy as np
import os
import ot
import matching
from matching.algorithms import galeshapley

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 设置使用的GPU卡，需在引入PyTorch之前

import argparse
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import load_deg_data, loss_func, f1_scores
from adapt import NuclearWassersteinDiscrepancy
from model import AttnCL, GradReverse

# 设置超参数
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='data/dblpv7.mat', help='Source Network')  # 设置源网络
parser.add_argument('--target', type=str, default='data/acmv9.mat', help='Target Network')  # 设置目标网络
parser.add_argument('--clf_type', type=str, default='multi-class', help='Training task')  # 分类任务类型，多类别或多标签
parser.add_argument('--train_ratio', type=float, default=1.0, help='Label rate of the source network')
parser.add_argument('--num_epoch', type=int, default=20000, help='Maximum training iteration')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--hidden_dim', type=str, default='1024,512', help='Hidden units dimension')
parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--num_class', type=int, default=5, help='Number of classes')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate')
parser.add_argument('--l2_w', type=float, default=1e-3, help='Weight of L2-norm regularization')
parser.add_argument('--seed', type=int, default=2023, help='Random seed')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--pred_hid', type=int, default=2048, help='Number of hidden units for the predictor')
parser.add_argument('--mad', type=float, default=0.9, help='Moving average decay for teacher network')
parser.add_argument('--num_centroids', type=int, default=5, help='The number of centroids for K-means clustering')
parser.add_argument('--num_kmeans', type=int, default=5, help='The number of kmeans clustering to be robust')
parser.add_argument('--cluster_num_iters', type=int, default=20)
parser.add_argument('--topk', type=int, default=5, help='The number of neighbors to search')
parser.add_argument('--gpu', type=str, default='2', help='The GPU card used')
args = parser.parse_args()

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
torch.set_num_threads(1)  # 设置Pytorch占用的CPU线程数，避免把服务器CPU资源全占了

# 设置程序运行位置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

# 读取并预处理数据
data_o, data_s, train_idx, test_idx, Y_s, Y_t, degree_o = load_deg_data(args)
feature_size = data_o.x.shape[1]
hidden_size = [int(x) for x in args.hidden_dim.strip().split(',')]
neighbor_idx = data_o.edge_index.to(device)  # 存在连接的节点对
neighbor_attr = torch.FloatTensor(np.arange(neighbor_idx.shape[1])).to(device)
edge_attr = torch.FloatTensor(np.arange(neighbor_idx.shape[1])).to(device)
source_idx = torch.LongTensor(np.arange(Y_s.shape[0])).to(device)
target_idx = torch.LongTensor(np.arange(Y_s.shape[0], Y_s.shape[0] + Y_t.shape[0])).to(device)

# 将数据存入cuda
data_o, data_s = data_o.to(device), data_s.to(device)
degree_o = degree_o.to(device)
train_idx, test_idx = train_idx.to(device), test_idx.to(device)

if args.clf_type == 'multi-label':
    Y_s, Y_t = torch.FloatTensor(Y_s).to(device), torch.FloatTensor(Y_t).to(device)
    clf_loss_func = nn.BCEWithLogitsLoss(reduction='none')  # 多标签分类损失函数
else:
    Y_s, Y_t = torch.argmax(torch.LongTensor(Y_s), dim=1).to(device), torch.argmax(torch.LongTensor(Y_t), dim=1).to(
        device)
    print(torch.max(Y_s), torch.max(Y_t))
    clf_loss_func = nn.CrossEntropyLoss()  # 多类别分类损失函数

Y_label = torch.cat((Y_s, Y_t), dim=0)
domain_label = np.vstack([np.tile([1., 0.], [Y_s.shape[0], 1]), np.tile([0., 1.], [Y_t.shape[0], 1])])
domain_label = torch.argmax(torch.FloatTensor(domain_label), dim=1).to(device)

# 定义模型和损失函数
model = AttnCL(args, feature_size, hidden_size, degree_o.shape[1]).to(device)
domain_loss_func = nn.CrossEntropyLoss()  # 对抗领域自适应损失
discrepancy = NuclearWassersteinDiscrepancy(model.node_classifier)

file = open(
    'results/test.txt',
    'a')

# 开启训练流程
for epoch in range(args.num_epoch):
    p = float(epoch) / args.num_epoch
    lr = args.lr / (1. + 10 * p) ** 0.75
    grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # 逐渐从0变到1
    GradReverse.rate = grl_lambda
    optimizer = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=args.l2_w / 2)

    model.train()
    optimizer.zero_grad()
    emb, teacher, student_project, ind, pred_logit, d_logit = model(data_o, data_s, [neighbor_idx, neighbor_attr],
                                                                 edge_attr, degree_o)

    contrast_loss = (loss_func(student_project[ind[0]], teacher[ind[1]].detach()) + loss_func(student_project[ind[1]],
                                                                                              teacher[ind[
                                                                                                  0]].detach())).mean()
    uf_loss = -discrepancy(emb, source_idx, target_idx)
    # 前期先用正常方式训练, 后期进行类别领域自适应和增量训练
    if epoch < 3000:
        if args.clf_type == 'multi-label':
            clf_loss = torch.sum(clf_loss_func(pred_logit[train_idx], Y_s[train_idx])) / torch.sum(Y_s[train_idx])
        else:
            clf_loss = clf_loss_func(pred_logit[train_idx], Y_s[train_idx])
        domain_loss = domain_loss_func(d_logit, domain_label)
    else:
        # 目标节点聚类, 并与源节点开展类别匹配
        src_emb = preprocessing.scale(emb[source_idx, :].detach().cpu().numpy())
        tgt_emb = preprocessing.scale(emb[test_idx, :].detach().cpu().numpy())
        lbl_s = Y_s.cpu().numpy()
        kmeans = faiss.Kmeans(args.emb_dim, args.num_class, niter=args.cluster_num_iters, gpu=False, seed=args.seed)
        kmeans.train(tgt_emb)
        _, I_kmeans = kmeans.index.search(tgt_emb, 1)
        cluster_label = I_kmeans[:, 0]  # 返回了每个目标节点所属的聚类类别
        clst_s = []  # 记录下源网络每个类别的节点 ID
        for lbl in range(args.num_class):
            clst_s.append(np.where(lbl_s == lbl))
        clst_t = []  # 记录下目标网络每个聚类的节点 ID
        for lbl in range(args.num_class):
            clst_t.append(np.where(cluster_label == lbl))

        # 计算源网络各类别与目标网络各聚类之间的最优传输距离
        src_tgt_dist = []
        for i in range(args.num_class):
            src_dist = []
            for j in range(args.num_class):
                src_node_emb = src_emb[clst_s[i]]
                tgt_node_emb = tgt_emb[clst_t[j]]
                a, b = np.ones((len(src_node_emb),)) / len(src_node_emb), np.ones((len(tgt_node_emb),)) / len(
                    tgt_node_emb)
                M = cdist(src_node_emb, tgt_node_emb, metric='euclidean')  # 计算类别矩阵和聚类矩阵的距离
                Gs = ot.sinkhorn2(a, b, M, 1e-1, numItermax=2000)  # 得到源类别和目标聚类之间的最优传输距离
                src_dist.append(Gs)
            src_tgt_dist.append(src_dist)
        src_tgt_dist = np.array(src_tgt_dist)

        # 开展源类别与目标聚类之间的稳定匹配
        suitor_preference, review_preference = {}, {}
        for i in range(len(src_tgt_dist)):
            suitor_preference[i] = list(np.argsort(src_tgt_dist[i, :]))
        for i in range(len(src_tgt_dist)):
            review_preference[i] = list(np.argsort(src_tgt_dist[:, i]))
        matching = galeshapley(suitor_preference, review_preference)  # 稳定匹配的结果

        # 筛选那些分类器预测与聚类预测一致的目标节点
        tgt_preds = torch.argmax(pred_logit[test_idx], dim=1)
        tgt_pred = tgt_preds.detach().cpu().numpy()
        conf_pred = [i for i in range(len(tgt_pred)) if matching[tgt_pred[i]] == cluster_label[i]]  # 分类预测与聚类预测一致的点
        # 将高置信度预测的目标节点放到分类损失和领域自适应损失中, 进行增强训练
        conf_pred = np.array(conf_pred)
        np.random.shuffle(conf_pred)
        sup_index = torch.LongTensor(conf_pred).to(device)
        sup_label = torch.LongTensor(tgt_pred[conf_pred]).to(device)

        # 根据预测补充的训练样本
        concat_index = torch.cat((train_idx, sup_index + len(train_idx)))
        concat_domain_label = torch.LongTensor(
            [0 for i in range(len(train_idx))] + [1 for i in range(len(sup_index))]).to(device)


        if args.clf_type == 'multi-label':
            clf_loss = torch.sum(clf_loss_func(pred_logit[train_idx], Y_s[train_idx])) / torch.sum(Y_s[train_idx])
        else:
            # clf_loss = clf_loss_func(pred_logit[concat_index], torch.cat((Y_label[train_idx], tgt_preds[sup_index])))
            clf_loss = clf_loss_func(pred_logit[train_idx], Y_s[train_idx])
        domain_loss = domain_loss_func(d_logit[concat_index], concat_domain_label)
    total_loss = domain_loss + clf_loss + contrast_loss + uf_loss
    total_loss.backward()
    optimizer.step()

    # 测试模型效果
    if epoch % 5 == 0:
        model.eval()
        emb, teacher, student_project, ind, pred_logit, d_logit = model(data_o, data_s, [neighbor_idx, neighbor_attr],
                                                                        edge_attr, degree_o)
        if args.clf_type == 'multi-label':
            # 多标签分类
            test_pred = F.sigmoid(pred_logit[test_idx])
            test_pred = test_pred.detach().cpu().numpy()
            y_t = Y_t.detach().cpu().numpy()
            micro_f1, macro_f1 = f1_scores(test_pred, y_t)
            print('Epoch: {}, Loss: {}, Micro F1: {}, Macro F1: {}'.format(epoch, total_loss, micro_f1, macro_f1))
        else:
            test_pred = torch.argmax(pred_logit[test_idx], dim=1)
            acc = torch.sum(test_pred == Y_t) / len(Y_t)
            print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, total_loss, acc))
            file.write('Epoch: ' + str(epoch) + '\t' + 'Loss: ' + str(total_loss) + '\t' + 'Accuracy: ' + str(acc) + '\n')

# 绘制嵌入向量的散点图
test_emb = emb[test_idx].detach().cpu().numpy()
test_y = torch.argmax(pred_logit[test_idx], dim=1).detach().cpu().numpy()
np.savetxt('results/test_emb.csv', test_emb, fmt='%f', delimiter=',')
np.savetxt('results/test_lbl.csv', test_y, fmt="%d", delimiter='\n')
test_x = TSNE(n_components=2).fit_transform(test_emb)
cmap = plt.cm.get_cmap('rainbow', 5)
plt.scatter(test_x[:, 0], test_x[:, 1], s=0.05, c=test_y, cmap=cmap)
# plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('results/test_vis.pdf')
plt.show()