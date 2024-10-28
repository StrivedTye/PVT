# Created by zenn at 2021/5/8
import torch
from torch import nn
from pointnet2.utils import pytorch_utils as pt_utils
from pointnet2.utils import pointnet2_utils

import torch.nn.functional as F
import numpy as np
try:
    from torch_scatter import scatter_add
except:
    print("torch_scatter is not installed.")


class BaseXCorr(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = pt_utils.SharedMLP([in_channel, hidden_channel, hidden_channel, hidden_channel], bn=True)
        self.fea_layer = (pt_utils.Seq(hidden_channel)
                          .conv1d(hidden_channel, bn=True)
                          .conv1d(out_channel, activation=None))


class P2B_XCorr(BaseXCorr):
    def __init__(self, feature_channel, hidden_channel, out_channel):
        mlp_in_channel = feature_channel + 4
        super().__init__(mlp_in_channel, hidden_channel, out_channel)

    def forward(self, template_feature, search_feature, template_xyz):
        """

        :param template_feature: B,f,M
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :return:
        """
        B = template_feature.size(0)
        f = template_feature.size(1)
        n1 = template_feature.size(2)
        n2 = search_feature.size(2)
        final_out_cla = self.cosine(template_feature.unsqueeze(-1).expand(B, f, n1, n2),
                                    search_feature.unsqueeze(2).expand(B, f, n1, n2))  # B,n1,n2

        fusion_feature = torch.cat(
            (final_out_cla.unsqueeze(1), template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B, 3, n1, n2)),
            dim=1)  # B,1+3,n1,n2

        fusion_feature = torch.cat((fusion_feature, template_feature.unsqueeze(-1).expand(B, f, n1, n2)),
                                   dim=1)  # B,1+3+f,n1,n2

        fusion_feature = self.mlp(fusion_feature)

        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])  # B, f, 1, n2
        fusion_feature = fusion_feature.squeeze(2)  # B, f, n2
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature


class BoxAwareXCorr(BaseXCorr):
    def __init__(self, feature_channel, hidden_channel, out_channel, k=8, use_search_bc=False, use_search_feature=False,
                 bc_channel=9):
        self.k = k
        self.use_search_bc = use_search_bc
        self.use_search_feature = use_search_feature
        mlp_in_channel = feature_channel + 3 + bc_channel

        if use_search_bc:
            mlp_in_channel += bc_channel
        if use_search_feature:
            mlp_in_channel += feature_channel
        super(BoxAwareXCorr, self).__init__(mlp_in_channel, hidden_channel, out_channel)

    def forward(self, template_feature, search_feature, template_xyz,
                search_xyz=None, template_bc=None, search_bc=None):
        """

        :param template_feature: B,f,M
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :param search_xyz: B.N,3
        :param template_bc: B,M,9
        :param search_bc: B.N,9
        :param args:
        :param kwargs:
        :return:
        """
        dist_matrix = torch.cdist(template_bc, search_bc)  # B, M, N
        template_xyz_feature_box = torch.cat([template_xyz.transpose(1, 2).contiguous(),
                                              template_bc.transpose(1, 2).contiguous(),
                                              template_feature], dim=1)
        # search_xyz_feature = torch.cat([search_xyz.transpose(1, 2).contiguous(), search_feature], dim=1)

        top_k_nearest_idx_b = torch.argsort(dist_matrix, dim=1)[:, :self.k, :]  # B, K, N
        top_k_nearest_idx_b = top_k_nearest_idx_b.transpose(1, 2).contiguous().int()  # B, N, K
        correspondences_b = pointnet2_utils.grouping_operation(template_xyz_feature_box,
                                                               top_k_nearest_idx_b)  # B,3+9+D,N,K
        if self.use_search_bc:
            search_bc_expand = search_bc.transpose(1, 2).unsqueeze(dim=-1).repeat(1, 1, 1, self.k)  # B,9,N,K
            correspondences_b = torch.cat([search_bc_expand, correspondences_b], dim=1)
        if self.use_search_feature:
            search_feature_expand = search_feature.unsqueeze(dim=-1).repeat(1, 1, 1, self.k)  # B,D,N,K
            correspondences_b = torch.cat([search_feature_expand, correspondences_b], dim=1)

        ## correspondences fusion head
        fusion_feature = self.mlp(correspondences_b)   # B,D,N,K
        fusion_feature, _ = torch.max(fusion_feature, dim=-1)  # B,D,N,1
        fusion_feature = self.fea_layer(fusion_feature.squeeze(dim=-1))  # B,D,N

        return fusion_feature


class PointVoxelXCorr(nn.Module):
    def __init__(self, num_levels=3, base_scale=0.25, resolution=3, truncate_k=128, num_knn=32, feat_channel=192):
        super(PointVoxelXCorr, self).__init__()

        self.truncate_k = truncate_k
        self.num_levels = num_levels
        self.resolution = resolution  # local resolution
        self.base_scale = base_scale  # search (base_scale * resolution)^3 cube

        self.vol_conv = nn.Sequential(
            nn.Conv1d((self.resolution ** 3) * self.num_levels, 128, 1),
            nn.GroupNorm(8, 128),
            nn.PReLU(),
            nn.Conv1d(128, feat_channel, 1)
        )

        self.knn = num_knn
        self.knn_conv = nn.Sequential(
            nn.Conv2d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.PReLU()
        )
        self.knn_out = nn.Conv1d(64, feat_channel, 1)

        self.knn_conv_f = nn.Sequential(
            nn.Conv2d(3+9+256, 256, 1),
            nn.GroupNorm(8, 256),
            nn.PReLU(),
            nn.Conv2d(256, 128, 1),
            nn.GroupNorm(8, 128),
            nn.PReLU(),
        )
        self.knn_out_f = nn.Conv1d(128, feat_channel, 1)

    def init_module(self, fmap1, fmap2, xyz1, xyz2, search_bc=None, template_bc=None):
        """

        :param fmap1: features of search region
        :param fmap2: features of template
        :param xyz1:  points of search region
        :param xyz2:  points of template
        :param search_bc:  box cloud of search region
        :param template_bc:  box cloud of template

        :return:
        """
        b, n_p, _ = xyz2.size()
        n_p1 = fmap1.size(2)

        if search_bc is not None and template_bc is not None:
            corr = torch.cdist(search_bc, template_bc)  # b, n_p1, n_p
        else:
            corr = self.calculate_corr(fmap1, fmap2)

        xyz2 = xyz2.view(b, 1, n_p, 3).expand(b, n_p1, n_p, 3)

        # corr_topk = torch.topk(corr.clone(), k=self.truncate_k, dim=2, sorted=True)
        # indx = corr_topk.indices.reshape(b, n_p1, self.truncate_k, 1).expand(b, n_p1, self.truncate_k, 3)
        # self.truncated_corr = corr_topk.values
        # self.truncate_xyz2 = torch.gather(xyz2, dim=2, index=indx)  # b, n_p1, k, 3

        self.truncated_corr = corr
        self.truncate_xyz2 = xyz2  # b, n_p1, k, 3

        self.ones_matrix = torch.ones_like(self.truncated_corr)

    def __call__(self, coords, coords2, fmap1=None, fmap2=None, search_bc=None, template_bc=None):
        self.init_module(fmap1, fmap2, coords, coords2)

        return self.get_voxel_feature(coords) \
               + self.get_knn_feature(coords, fmap1, fmap2, search_bc, template_bc)

        # return torch.cat([self.get_voxel_feature(coords),
        #                   self.get_knn_feature(coords, fmap1, fmap2, search_bc, template_bc)],
        #                  dim=1)

        # return self.get_knn_feature(coords, fmap1, fmap2, search_bc, template_bc)
        # return self.get_voxel_feature(coords)

        # return self.get_knn_feature_fusion(coords, fmap1, fmap2, search_bc, template_bc)

        # return self.get_voxel_feature(coords) \
        #        + self.get_knn_feature_fusion(coords, fmap1, fmap2, search_bc, template_bc) #gf

        # return self.get_knn_feature(coords) \
        #        + self.get_knn_feature_fusion(coords, fmap1, fmap2, search_bc, template_bc) #gf2

    def get_voxel_feature(self, coords):
        '''

        :param coords: search region
        :return:
        '''

        b, n_p, _ = coords.size()
        corr_feature = []
        for i in range(self.num_levels):
            with torch.no_grad():
                r = self.base_scale * (2 ** i)
                dis_voxel = torch.round((self.truncate_xyz2 - coords.unsqueeze(dim=-2)) / r)  # [b, n_s, n_t, 3]
                valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.resolution / 2)).all(dim=-1)   # [b, n_s, n_t]
                dis_voxel = dis_voxel - (-1)
                cube_idx = dis_voxel[:, :, :, 0] * (self.resolution ** 2) + \
                           dis_voxel[:, :, :, 1] * self.resolution + dis_voxel[:, :, :, 2]
                cube_idx_scatter = cube_idx.type(torch.int64) * valid_scatter

                valid_scatter = valid_scatter.detach()
                cube_idx_scatter = cube_idx_scatter.detach()

            corr_add = scatter_add(self.truncated_corr * valid_scatter, cube_idx_scatter)
            corr_cnt = torch.clamp(scatter_add(self.ones_matrix * valid_scatter, cube_idx_scatter), 1, n_p)
            corr = corr_add / corr_cnt
            if corr.shape[-1] != self.resolution ** 3:
                repair = torch.zeros([b, n_p, self.resolution ** 3 - corr.shape[-1]], device=coords.device)
                corr = torch.cat([corr, repair], dim=-1)

            corr_feature.append(corr.transpose(1, 2).contiguous())

        return self.vol_conv(torch.cat(corr_feature, dim=1))

    def get_knn_feature(self, coords, fmap1=None, fmap2=None, search_bc=None, template_bc=None):
        """

        :param coords: the points of search region
        :param fmap1: features of search region
        :param fmap2: features of template
        :return:
        """

        b, n_p, _ = coords.size()

        dist = self.truncate_xyz2 - coords.view(b, n_p, 1, 3)
        dist = torch.sum(dist ** 2, dim=-1)  # b, 1024, 512
        # dist = torch.cdist(search_bc, template_bc)

        neighbors = torch.topk(-dist, k=self.knn, dim=2).indices

        knn_corr = torch.gather(self.truncated_corr.view(b * n_p, -1), dim=1,
                                index=neighbors.reshape(b * n_p, self.knn)).reshape(b, 1, n_p, self.knn)

        neighbors_xyz = neighbors.view(b, n_p, self.knn, 1).expand(b, n_p, self.knn, 3)
        knn_xyz = torch.gather(self.truncate_xyz2, dim=2, index=neighbors_xyz).permute(0, 3, 1, 2).contiguous()
        knn_xyz = knn_xyz - coords.transpose(1, 2).reshape(b, 3, n_p, 1)

        knn_feature = self.knn_conv(torch.cat([knn_corr, knn_xyz], dim=1))
        knn_feature = torch.max(knn_feature, dim=3)[0]
        return self.knn_out(knn_feature)

    def get_knn_feature_fusion(self, coords, fmap1=None, fmap2=None, search_bc=None, template_bc=None):

        b, n_p, _ = coords.size()

        dist = torch.cdist(search_bc, template_bc)
        neighbors = torch.topk(-dist, k=self.knn, dim=2).indices

        # knn_corr = torch.gather(self.truncated_corr.view(b * n_p, -1), dim=1,
        #                         index=neighbors.reshape(b * n_p, self.knn)).reshape(b, 1, n_p, self.knn)

        # template features
        neighbors_fmap = neighbors.unsqueeze(1).expand(b, fmap2.size(1), n_p, self.knn)
        knn_fmap2 = fmap2.unsqueeze(2).repeat(1, 1, n_p, 1)
        knn_fmap2 = torch.gather(knn_fmap2, dim=3, index=neighbors_fmap)  #[b,c,n,k]

        # template coordinates
        neighbors_xyz = neighbors.view(b, n_p, self.knn, 1).expand(b, n_p, self.knn, 3)
        knn_xyz2 = torch.gather(self.truncate_xyz2, dim=2, index=neighbors_xyz).permute(0, 3, 1, 2).contiguous()

        # template box cloud
        template_bc = template_bc.unsqueeze(1).repeat(1, n_p, 1, 1)
        neighbors_bc = neighbors.view(b, n_p, self.knn, 1).expand(b, n_p, self.knn, template_bc.size(-1))
        template_bc = torch.gather(template_bc, dim=2, index=neighbors_bc).permute(0, 3, 1, 2).contiguous()

        # search region's features
        # knn_fmap1 = fmap1.unsqueeze(-1).repeat(1, 1, 1, self.knn)
        # knn_feature = torch.cat([knn_fmap1, knn_xyz2, template_bc, knn_fmap2], dim=1)

        knn_feature = torch.cat([knn_xyz2, template_bc, knn_fmap2], dim=1)

        knn_feature = self.knn_conv_f(knn_feature)
        knn_feature = torch.max(knn_feature, dim=3)[0]
        return self.knn_out_f(knn_feature)

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        return corr


class OTXCorr(nn.Module):
    def __init__(self, solver_iter=100, knn=32):
        super(OTXCorr, self).__init__()
        self.solver_iter = solver_iter
        self.knn = knn
        self.knn_conv = pt_utils.SharedMLP([1+3+9+256, 128, 256], bn=True)
        self.knn_out = nn.Conv1d(256, 32, 1)

        # self.u_pred = (
        #     pt_utils.Seq(256)
        #     .conv1d(256, bn=True)
        #     .conv1d(1)
        # )

    def __call__(self, fmap1, fmap2, xyz1, xyz2, bc1=None, bc2=None):
        return self.get_knn_feature_fusion(fmap1, fmap2, xyz1, xyz2, bc1, bc2) #[B, C, np1]

    def get_knn_feature_fusion(self, fmap1, fmap2, xyz1, xyz2, bc1=None, bc2=None):

        b, c1, n_p1 = fmap1.size()
        b, c2, n_p2 = fmap2.size()  # template

        T, u = self.calc_similarity(fmap1, fmap2, xyz1, xyz2)
        T = torch.clamp(T, 1e-7, 1)

        neighbors = torch.topk(T, k=self.knn, dim=2).indices

        knn_corr = torch.gather(T.view(b * n_p1, n_p2), dim=1,
                                index=neighbors.reshape(b * n_p1, self.knn)).reshape(b, 1, n_p1, self.knn)

        # neighbors_fmap = neighbors.unsqueeze(1).expand(b, c2, n_p1, self.knn)
        # knn_fmap2 = fmap2.unsqueeze(2).expand(b, c2, n_p1, n_p2)
        # knn_fmap2 = torch.gather(knn_fmap2, dim=3, index=neighbors_fmap) #[b,c,n,k]

        knn_fmap1 = fmap1.unsqueeze(-1).expand(b, c1, n_p1, self.knn)
        clue2 = torch.cat([xyz2.transpose(1, 2).contiguous(),
                           bc2.transpose(1, 2).contiguous(),
                           fmap2], dim=1) #[b, 3+9+256, n]
        neighbors_clue2 = neighbors.unsqueeze(1).expand(b, clue2.size(1), n_p1, self.knn)
        clue2 = clue2.unsqueeze(2).expand(b, clue2.size(1), n_p1, n_p2)
        knn_clue2 = torch.gather(clue2, dim=3, index=neighbors_clue2)

        knn_feature = self.knn_conv(torch.cat([knn_corr, knn_clue2], dim=1))
        # knn_feature = self.knn_conv(torch.cat([knn_corr, knn_clue2, knn_fmap1], dim=1))

        knn_feature = torch.max(knn_feature, dim=3)[0]
        return self.knn_out(knn_feature)

    def calc_similarity(self, search, template, search_points, template_points, use_uniform=False):

        _, _, n_ps = search.size()
        B, _, n_pt = template.size()

        search_norm = F.normalize(search, p=2, dim=1) #[B, C, np1]
        template_norm = F.normalize(template, p=2, dim=1)
        f_sim = torch.matmul(search_norm.transpose(1, 2), template_norm) #[B, np1, np2]
        g_sim = torch.cdist(search_points, template_points) # [B, np1, np2]
        # f_sim = self.calculate_corr(search, template)
        cost = (1 - f_sim) + 0.1 * g_sim
        cost = torch.clamp(cost, 0, 1)
        K = torch.exp(-cost / 0.1)

        if use_uniform:
            u = torch.zeros(B, n_ps, dtype=cost.dtype, device=cost.device).fill_(1. / n_ps) # marginal distribution of search points
            v = torch.zeros(B, n_pt, dtype=cost.dtype, device=cost.device).fill_(1. / n_pt)
        else:
            # search_avg = search.mean(2).unsqueeze(1) #[B, 1, C]
            # att = F.relu(torch.matmul(search_avg, template)).squeeze(1) # [B, n_pt]
            # v = att / (att.sum(dim=1, keepdim=True) + 1e-6)
            v = torch.zeros(B, n_pt, dtype=cost.dtype, device=cost.device).fill_(1. / n_pt)

            template_avg = template.mean(2).unsqueeze(1)
            att = F.relu(torch.matmul(template_avg, search)).squeeze(1) # [B, n_ps]
            u = att / (att.sum(dim=1, keepdim=True) + 1e-6)
            # uu = self.u_pred(search).squeeze(1)
        T = self.Sinkhorn(K, u, v)
        return T, f_sim

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-1
        for _ in range(self.solver_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1) #[B, np1]
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1) #[B, np2]
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        return corr

