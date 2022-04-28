import torch.nn as nn
import torch.nn.functional as F
from .model import farthest_point_sample

class Loss(nn.Module):
    def __init__(self):
        """
        初始化
        """
        super(Loss, self).__init__()

    def stats_boundary_loss(self, coords, feats):
        # TODO: cal the L_cbl
        pass

    def stats_ordinary_loss(self, labels, indexs, preds, num_classes):

        pred1, pred2, pred3, pred4, pred5 = preds
        label2, label3, label4, label5 = labels[indexs]

        loss1 = F.cross_entropy(pred1.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        loss2 = F.cross_entropy(pred2.contiguous().view(-1, num_classes), label2.contiguous().view(-1))
        loss3 = F.cross_entropy(pred3.contiguous().view(-1, num_classes), label3.contiguous().view(-1))
        loss4 = F.cross_entropy(pred4.contiguous().view(-1, num_classes), label4.contiguous().view(-1))
        loss5 = F.cross_entropy(pred5.contiguous().view(-1, num_classes), label5.contiguous().view(-1))

        return loss1 + loss2 + loss3 + loss4 + loss5

    def forward(self, points, labels, indexs, output, preds, coords, feats, num_classes=13, alpha=0.1, beta=0.1):
        """
        :param points: 输入的点集, B x N x 6
        :param labels: 输入的标签, B x N x 1
        :param indexs: 每次FPS下采样所采点的集合, [index2, index3, index4, index5]
        :param output: 模型预测结果, B x N x 13
        :param preds: 模型5个stage的预测标签, [pred1, pred2, pred3, pred4, pred5]
        :param coords: 5个stage中点的坐标, [xyz1, xyz2, xyz3, xyz4, xyz5]
        :param feats: 5个stage中特征, [x1, x2, x3, x4, x5]
        :param num_classes: 语义标签种类
        :param alpha: 边界损失权重
        :param beta: 普通点损失权重
        :return: loss
        """
        output_loss = F.cross_entropy(output.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        boundary_loss = self.stats_boundary_loss(coords, feats)
        ordinary_loss = self.stats_ordinary_loss(labels, indexs, preds, num_classes)

        loss = output_loss + alpha * boundary_loss + beta * ordinary_loss
        return loss