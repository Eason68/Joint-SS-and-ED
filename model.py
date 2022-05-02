import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import farthest_point_sample, knn_point, index_points, square_distance

class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        """
        Give xyz[B, N, 3] and feat[B, N, D], return new_xyz[B, N/4, 3] and new_feat[B, N/4, K, D+3+D]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, self.groups).long()  # [B, S]
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
        new_points = index_points(points, fps_idx)  # [B, S, D]

        idx = knn_point(self.kneighbors, xyz, new_xyz)  # [B, S, K]
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, S, K, 3]
        grouped_points = index_points(points, idx)  # [B, S, K, D]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, S, K, D+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points, fps_idx


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvBNReLU1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True):
        super(ConvBNReLURes1D, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            nn.ReLU(inplace=True)
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return F.relu(self.net2(self.net1(x)) + x, inplace=True)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True, use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, mlp=[]):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups, res_expansion=res_expansion, bias=bias)

        # TODO: generate prediction
        self.prediction = nn.ModuleList()
        for channel in mlp:
            self.prediction.append(ConvBNReLU1D(out_channel, channel, 1, bias=bias))
            self.prediction.append(nn.Dropout(0.5))
            out_channel = channel
        self.prediction.append(ConvBNReLU1D(out_channel, 13, 1, bias=bias))

    # TODO: add pred
    def forward(self, xyz1, xyz2, points1, points2, last_pred):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
            last_pred: [B, S, 13]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # if S == 1:
        #     interpolated_points = points2.repeat(1, N, 1)
        # else:
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        # TODO: upsample pred
        interpolated_preds = torch.sum(index_points(last_pred, idx) * weight.view(B, N, 3, 1), dim=2)  # [B, N, 13]

        # if points1 is not None:
        #     points1 = points1.permute(0, 2, 1)
        #     new_points = torch.cat([points1, interpolated_points], dim=-1)
        # else:
        #     new_points = interpolated_points

        # TODO: concat pred branch to main
        new_points =torch.cat((points1, interpolated_points, interpolated_preds), dim=-1)

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)

        # TODO: generate the pred
        feat = new_points
        for i in range(len(self.prediction) - 1):
            feat = self.prediction[i](feat)

        pred = self.prediction[-1](feat)

        # TODO: return the pred
        return new_points, feat.permute(0, 2, 1), pred.permute(0, 2, 1)




class MyModel(nn.Module):
    def __init__(self, num_classes=13, points=4096, in_channel=9, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[1024, 512, 256, 128, 128], de_blocks=[2, 2, 2, 2],
                 gmp_dim=64,cls_dim=64, **kwargs):
        super(MyModel, self).__init__()
        self.stages = len(pre_blocks)
        self.points = points
        self.embedding = ConvBNReLU1D(in_channel, embed_dim, bias=bias)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups, res_expansion=res_expansion, bias=bias)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)


        # TODO: build feat5 and pred5
        self.feature = nn.Sequential(ConvBNReLU1D(1024, 256), nn.Dropout(), ConvBNReLU1D(256, 64), nn.Dropout(), ConvBNReLU1D(64, 32))
        self.predict = nn.Sequential(nn.Dropout(), ConvBNReLU1D(32, 13), nn.LogSoftmax(dim=1))

        ### Building Decoder #####
        # self.decode_list = nn.ModuleList()
        # en_dims.reverse()
        # de_dims.insert(0, en_dims[-1])
        # assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        # for i in range(len(en_dims)-1):
        #     self.decode_list.append(
        #         PointNetFeaturePropagation(de_dims[i] + en_dims[i+1] + num_classes, de_dims[i+1],
        #                                    blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
        #                                    bias=bias, activation=activation))
        self.fp4 = PointNetFeaturePropagation(de_dims[0] + en_dims[3] + num_classes, de_dims[1], de_blocks[0], mlp=[256, 128, 64, 32])
        self.fp3 = PointNetFeaturePropagation(de_dims[1] + en_dims[2] + num_classes, de_dims[2], de_blocks[1], mlp=[128, 64, 32])
        self.fp2 = PointNetFeaturePropagation(de_dims[2] + en_dims[1] + num_classes, de_dims[3], de_blocks[2], mlp=[64, 32])
        self.fp1 = PointNetFeaturePropagation(de_dims[3] + en_dims[0] + num_classes, de_dims[4], de_blocks[3], mlp=[64, 32])


        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=bias)

        # classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim + de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        coord = x[:, 0:3, :].permute(0, 2, 1)
        # x = torch.cat([x,norm_plt],dim=1)
        x = self.embedding(x)  # B,D,N

        coords = [coord]  # [B, N, 3]
        x_list = [x]  # [B, D, N]
        indexs = []

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            coord, x, index = self.local_grouper_list[i](coord, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            coords.append(coord)
            x_list.append(x)
            indexs.append(index)

        # here is the decoder
        # xyz_list.reverse()
        # x_list.reverse()
        x5 = x_list[-1]  # [B, 1024, N/256]

        # for i in range(len(self.decode_list)):
        #     x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1], x)

        # TODO: decoder
        feat = self.feature(x5)
        feat5 = feat.permute(0, 2, 1)
        pred5 = self.predict(feat).permute(0, 2, 1) # [B, N/256, 13]
        x4, feat4, pred4 = self.fp4(coords[3], coords[4], x_list[3], x5, pred5)  # [B, 512, N/64], [B, N/64, 13]
        x3, feat3, pred3 = self.fp3(coords[2], coords[3], x_list[2], x4, pred4)  # [B, 256, N/16], [B, N/16, 13]
        x2, feat2, pred2 = self.fp2(coords[1], coords[2], x_list[1], x3, pred3)  # [B, 128, N/4],  [B, N/4, 13]
        x1, feat1, pred1 = self.fp1(coords[0], coords[1], x_list[0], x2, pred2)  # [B, 128, N],    [B, N, 13]

        feats = [feat1, feat2, feat3, feat4, feat5]
        preds = [pred1, pred2, pred3, pred4, pred5]



        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]

        output = torch.cat([x1, global_context.repeat([1, 1, x1.shape[-1]])], dim=1)
        output = self.classifier(output)
        # x = F.log_softmax(x, dim=1)
        output = output.permute(0, 2, 1)

        return output, coords, feats, preds, indexs


# def MyModel(num_classes=13, **kwargs) -> MyModel:
#     return MyModel(num_classes=num_classes, points=4096, embed_dim=64, groups=1, res_expansion=1.0,
#                  activation="relu", bias=True, use_xyz=True, normalize="anchor",
#                  dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
#                  k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
#                  de_dims=[1024, 512, 256, 128, 128], de_blocks=[4,4,4,4],
#                  gmp_dim=64,cls_dim=64, **kwargs)
#
#
# if __name__ == '__main__':
#     data = torch.rand(2, 6, 1024)
#     print("testing model ...")
#     model = MyModel()
#     out, xyzs, xs, preds, indexs = model(data)  # [2,2048,13]
#     print("model output:", out.shape)
#     print()
#     for xyz in xyzs:
#         print(xyz.shape)
#     print()
#     for x in xs:
#         print(x.shape)
#     print()
#     for pred in preds:
#         print(pred.shape)
#     print()
#     for index in indexs:
#         print(index.shape)
#     print()