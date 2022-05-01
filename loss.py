import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import knn_point, index_points

class Loss(nn.Module):
    def __init__(self):
        """
        初始化
        """
        super(Loss, self).__init__()
        self.nsamples = [64, 32, 16, 8, 4]
        self.temperature = 1.0

    def stats_boundary_loss(self, coords, labels, feats, num_classes):
        """
        边界损失函数 L_cbl
        :param coords: 原始坐标和4次下采样的坐标，用于计算邻域
        :param labels: 原始标签
        :param feats: decoder中由points经MLP生成，32通道
        :param num_classes: 语义类别
        :return: loss
        """
        loss = torch.tensor(.0).to(labels.device)
        labels = F.one_hot(labels, num_classes)  # [B, N, 13]
        for i, (coord, feat) in enumerate(zip(coords, feats)):

            # 生成场景标签
            if i == 0:
                probability_labels = labels
            else:
                kneighbors = i * 4  # K个邻居, 1, 4, 16, 64, 256
                neighbor_indexs = knn_point(kneighbors, coords[0], coord)  # [B, S, K], S: number of coords[i]
                probability_labels = index_points(labels, neighbor_indexs)  # [B, S, K, 13]
                probability_labels = probability_labels.float().mean(dim=-2)  # [B, S, 13]

            # 对标签和特征选择nsample邻域
            nsample = self.nsamples[i]
            neighbor_indexs = knn_point(nsample, coord, coord)  # [B, S, nsample], S: number of coords[i]
            neighbor_labels = index_points(probability_labels, neighbor_indexs)  # [B, S, K, 13]
            neighbor_feats = index_points(feat, neighbor_indexs)  # [B, S, K, C]

            # 查找边界点
            center_label = torch.argmax(torch.unsqueeze(probability_labels, -2), -1)  # [B, S, 1, 13] -> [B, S, 1]
            neighbor_labels = torch.argmax(neighbor_labels, -1)  # [B, S, K]
            mask = center_label == neighbor_labels  # [B, S, K], bool
            point_mask = torch.sum(mask.int(), -1)  # [B, S], 每个点的邻居有多少是和自己标签相同的
            point_mask = (point_mask > 0) & (point_mask < nsample)  # [B, S], bool

            # 没有边界点则直接进入下一个stage
            if not torch.any(point_mask):
                continue

            # 获得边界点特征
            mask = mask[point_mask]  # [B, sub_S, K]
            feat = feat[point_mask]  # [B, sub_S, C]
            neighbor_feats = neighbor_feats[point_mask]  # [B, sub_S, K, C]

            # 计算loss
            dist = torch.unsqueeze(feat, -2) - neighbor_feats  # [B, sub_S, 1, C] - [B, sub_S, K, C]
            dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + 1e-6)  # [B, sub_S, K, C] -> [B, sub_S, K]
            dist = - dist
            dist = dist - torch.max(dist, -1, keepdim=True)[0]
            dist = dist / self.temperature

            exp = torch.exp(dist)  # [B, sub_S, K]
            pos = torch.sum(exp * mask.float())
            neg = torch.sum(exp)

            loss = loss - torch.log(pos / (neg + 1e-6))

        return loss

    def stats_ordinary_loss(self, labels, indexs, preds, num_classes):
        """
        计算从decoder输出的5个预测结果preds的损失
        :param labels: 原始标签
        :param indexs: encoder每个stage下采样的点的下标, [B, N/4], ... , [B, N/256]
        :param preds: decoder每个stage生成的预测标签, [B, N, 13], ... , [B, N/64, 13], [B, N/256, 13]
        :param num_classes: 语义类别，默认13
        :return: loss
        """
        pred1, pred2, pred3, pred4, pred5 = preds
        label2, label3, label4, label5 = [labels.gather(dim=1, index=indexs[i]) for i in range(len(indexs))]

        loss1 = F.cross_entropy(pred1.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        loss2 = F.cross_entropy(pred2.contiguous().view(-1, num_classes), label2.contiguous().view(-1))
        loss3 = F.cross_entropy(pred3.contiguous().view(-1, num_classes), label3.contiguous().view(-1))
        loss4 = F.cross_entropy(pred4.contiguous().view(-1, num_classes), label4.contiguous().view(-1))
        loss5 = F.cross_entropy(pred5.contiguous().view(-1, num_classes), label5.contiguous().view(-1))

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        return loss

    def forward(self, labels, indexs, output, preds, coords, feats, num_classes=13, alpha=0.05, beta=0.05):
        """
        :param labels: 输入的标签, B x N x 1
        :param indexs: 每次FPS下采样所采点的集合, [index2, index3, index4, index5]
        :param output: 模型预测结果, B x N x 13
        :param preds: 模型5个stage的预测标签, [pred1, pred2, pred3, pred4, pred5]
        :param coords: 5个stage中点的坐标, [xyz1, xyz2, xyz3, xyz4, xyz5]
        :param feats: 5个stage中特征, [feat1, feat2, feat3, feat4, feat5]
        :param num_classes: 语义标签种类
        :param alpha: 边界损失权重
        :param beta: 普通点损失权重
        :return: loss
        """
        output_loss = F.cross_entropy(output.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        boundary_loss = self.stats_boundary_loss(coords, labels, feats, num_classes)
        ordinary_loss = self.stats_ordinary_loss(labels, indexs, preds, num_classes)

        loss = output_loss + alpha * boundary_loss + beta * ordinary_loss
        return loss
