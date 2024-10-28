""" 
Created by tye at 2021/11/10
"""

import torch
from torch import nn
from pointnet2.utils import pytorch_utils as pt_utils
from roiaware_pool3d.roiaware_pool3d_utils import RoIAwarePool3d


class SequentialDescent(nn.Module):

    def __init__(self, num_voxels_roi, max_pts_each_voxel=128, learn_R=True, num_R=3):
        super().__init__()

        self.roiaware_pool3d_layer = RoIAwarePool3d(out_size=num_voxels_roi,
                                                    max_pts_each_voxel=max_pts_each_voxel)

        self.proposal_feat = (pt_utils.Seq(num_voxels_roi ** 3 * 256)
                              .conv1d(256, bn=True)
                              .dropout(0.3)
                              .conv1d(256, bn=True)
                              )

        self.similar_feat = (pt_utils.Seq(256)
                             .conv1d(256, bn=True)
                             )

        self.reg_feat = (pt_utils.Seq(256)
                         .conv1d(256, bn=True)
                         )

        self.learn_R = learn_R
        self.num_R = num_R
        if self.learn_R:
            self.R_module = nn.ModuleList()
            for i in range(self.num_R):
                self.R_module.append(
                    (pt_utils.Seq(256)
                     .conv1d(256, bn=True)
                     .conv1d(256, bn=True)
                     .conv1d(3+1, activation=None)
                     )
                )
        else:
            self.R_module = nn.ParameterList(
                [nn.Parameter(torch.zeros([256, 4]), requires_grad=False) for i in range(self.num_R)]
            )

            self.first = True

        self.similar_pred = (pt_utils.Seq(512)
                             .conv1d(256, bn=True)
                             .conv1d(256, bn=True)
                             .conv1d(1, activation=None))

    def forward(self, template_xyz, search_xyz, template_feat, search_feat,
                proposals, search_box_input, template_box, search_gt_state=None, train=True):

        num_prop_vote = proposals.shape[1]
        search_box = template_box.repeat(1, num_prop_vote, 1)
        search_box[:, :, [0, 1, 2, 6]] = proposals[:, :, :4]

        if not self.learn_R:
            search_box = torch.cat([search_box_input, search_box], dim=1)

        search_box_list = [search_box[:, -num_prop_vote:, :]]
        score_list, delta_p_list = [], []
        nseed_per_frame = search_box.shape[1]

        # crop feature inside box from 'template', shape: [B, out_x, out_y, out_z, C]
        # roiPool(PointRcnn CVPR2019), RoIAwarePool(partA2), RoIGrid(PVRCNN) or pointPool(STD ICCV2019)??
        template_pooled_feat = self.roi_aware_pool(template_xyz, template_box, template_feat)
        search_pooled_feat = self.roi_aware_pool(search_xyz, search_box, search_feat)

        template_pooled_feat_f = torch.flatten(template_pooled_feat, 1).unsqueeze(-1) # [B*1, C*N, 1]
        search_pooled_feat_f = torch.flatten(search_pooled_feat, 1).unsqueeze(-1)  # [B*m, C*N, 1]

        T_proposal_feat = self.proposal_feat(template_pooled_feat_f)  # [B*1, 256, 1]
        T_similar_feat = self.similar_feat(T_proposal_feat)
        T_reg_feat = self.reg_feat(T_proposal_feat)

        T_similar_feat = T_similar_feat.unsqueeze(1).repeat(1, nseed_per_frame, 1, 1).flatten(0, 1)  # [B*m, 256, 1]
        T_reg_feat = T_reg_feat.unsqueeze(1).repeat(1, nseed_per_frame, 1, 1).flatten(0, 1)  # [B*m, 256, 1]

        for idx in range(self.num_R):
            search_pooled_feat_f, search_box, score, delta_p = self.progressive(idx, nseed_per_frame,
                                                                                search_pooled_feat_f, search_box,
                                                                                search_xyz, search_feat,
                                                                                T_similar_feat, T_reg_feat,
                                                                                search_gt_state, train)

            search_box_list.append(search_box[:, -num_prop_vote:, :])  # [[B, m, 7], ...]
            score_list.append(score[:, -num_prop_vote:, :])  # [[B, m, 1], ...]
            delta_p_list.append(delta_p[:, -num_prop_vote:, :])  # [[B, m, 4], ...]

        # calculating similarity for last step
        C_proposal_feat = self.proposal_feat(search_pooled_feat_f)  # [B*m, 256, 1]
        C_similar_feat = self.similar_feat(C_proposal_feat)  # [B*m, 256, 1]
        fused_similar_feat = torch.cat((C_similar_feat, T_similar_feat), dim=1)
        score = self.similar_pred(fused_similar_feat)
        score = score.view(-1, nseed_per_frame, 1)

        score_list.append(score[:, -num_prop_vote:, :])
        search_box_all = torch.cat(search_box_list, dim=1)  # [B, m*(num_R+1), 7]
        score_all = torch.cat(score_list, dim=1)  # [B, m*(num_R+1), 1]
        delta_p_all = torch.cat(delta_p_list, dim=1)  # [B, m*num_R, 4]
        self.first = False

        return search_box_all, delta_p_all, score_all

    def progressive(self, idx, num_seed,
                    pooled_feat_f, box,
                    search_xyz, search_feat,
                    T_similar_feat, T_reg_feat,
                    search_gt_state, train=True):

        # loop
        C_proposal_feat = self.proposal_feat(pooled_feat_f)  # [B*m, 256, 1]
        C_similar_feat = self.similar_feat(C_proposal_feat)
        C_reg_feat = self.reg_feat(C_proposal_feat)

        fused_similar_feat = torch.cat((C_similar_feat, T_similar_feat), dim=1)
        score = self.similar_pred(fused_similar_feat)
        score = score.view(-1, num_seed, 1)

        # regression
        # step 0->1
        fused_feat = C_reg_feat - T_reg_feat
        if not self.learn_R:
            if search_gt_state is not None:
                # if self.first:
                #     R, P = self.recursive_LS(fused_feat, box, search_gt_state, first=self.first)
                #     self.R_module[idx].copy_(R)
                #     self.P_module[idx].copy_(P)
                # else:
                #     R, P = self.recursive_LS(fused_feat, box, search_gt_state, self.R_module[idx], self.P_module[idx], first=self.first)
                #     self.R_module[idx].copy_(R)
                #     self.P_module[idx].copy_(P)
                # delta_p = fused_feat.squeeze(-1).matmul(R)
                # delta_p = delta_p.view(box.size(0), num_seed, -1)# [B, m, 4]

                R = self.cal_R_closed_form(fused_feat, box, search_gt_state, train)

                if not train:
                    self.R_module[idx].copy_(R)

                # delta_p = fused_feat.squeeze(-1).view(box.size(0), num_seed, -1).bmm(R)  # [B, m, 4]
                delta_p = fused_feat.squeeze(-1).matmul(R)
                delta_p = delta_p.view(box.size(0), num_seed, -1)  # [B, m, 4]
            else:
                # delta_p = fused_feat.squeeze(-1).view(box.size(0), num_seed, -1).bmm(self.R_module[idx]) #[B, m, 4]
                delta_p = fused_feat.squeeze(-1).matmul(self.R_module[idx])
                delta_p = delta_p.view(box.size(0), num_seed, -1)  # [B, m, 4]

            search_pooled_feat_f, search_box = self.get_pooled_feat(delta_p, box, search_xyz, search_feat)
            return search_pooled_feat_f, search_box, score, delta_p

        else:
            delta_p = self.R_module[idx](fused_feat)  # [B*m, 4, 1]
            delta_p = delta_p.transpose(1, 2).contiguous()
            delta_p = delta_p.view(-1, num_seed, 1, 4).squeeze(2)  # [B, m, 4]

            search_pooled_feat_f, search_box = self.get_pooled_feat(delta_p, box, search_xyz, search_feat)
            return search_pooled_feat_f, search_box, score, delta_p

    def roi_aware_pool(self, coord, roi, feat):
        """
        :param coord: coordinate [B, N, 3]
        :param roi:  bounding box [B, m, 7]
        :param feat: point-wise feature [B, C, N]
        :return: pooled_feature [B*m, voxel_num, voxel_num, voxel_num, C]
        """

        feat = feat.transpose(1, 2).contiguous()

        batch_size = coord.shape[0]
        pooled_feat_list = []
        for i in range(batch_size):
            cur_coord = coord[i]
            cur_roi = roi[i].contiguous()
            cur_feat = feat[i]
            pooled_feat = self.roiaware_pool3d_layer(cur_roi, cur_coord, cur_feat, pool_method='max')  # [m, 6, 6, 6, C]
            pooled_feat_list.append(pooled_feat)
        pooled_features = torch.cat(pooled_feat_list, dim=0)  # [B*m, ...]
        return pooled_features  # [B*m, 6, 6, 6, C]

    def cal_R_closed_form(self, X, search_box, search_gt_state, train=True):
        """
        (I+X'X)^(-1)X'Y ==> X'(I+XX')^(-1)Y
        X: fused_feat, i.e. C_reg_feat - T_reg_feat, of shape [B*m, 256, 1]
        search_gt_state: ground-truth of seach region [B, 1, 7]
        search_box: candidate boxes, of shape [B, m, 7]
        """

        X = X.view(search_box.size(0), search_box.size(1), -1) #[B, m, 256]
        search_gt_state = search_gt_state.repeat(1, search_box.size(1), 1)  # [B, m, 7]
        Y = search_gt_state - search_box  # [B, m, 7]
        Y = Y[:, :, [0, 1, 2, 6]] #[B, m, 4]

        index_for_RR = 4  # if train else 64 #int(search_box.size(1)/2)

        # X = X[:, 0:index_for_RR, :]
        # Y = Y[:, 0:index_for_RR, :]
        # Xt = X.transpose(1, 2).contiguous()
        # H = X.bmm(Xt) + torch.eye(index_for_RR).to(X) * 0.1
        # H_inv = torch.inverse(H) #[B, m, m]
        # pinv = Xt.bmm(H_inv) # [B, 256, m]
        # R = pinv.bmm(Y) # [B, 256, 4]

        X = X[:, 0:index_for_RR, :].contiguous().view(-1, X.size(-1)) #[n, 256]
        Y = Y[:, 0:index_for_RR, :].contiguous().view(-1, Y.size(-1)) #[n, 4]

        # (I+X'X)^(-1)X'T
        # Xt = X.transpose(0, 1).contiguous()
        # P = Xt.mm(X) + torch.eye(X.size(-1)).to(X) * 0.1
        # R = torch.inverse(P).mm(Xt).mm(Y)

        # X'(I+XX')^(-1)Y
        Xt = X.transpose(0, 1).contiguous()
        P = X.mm(Xt) + torch.eye(X.size(0)).to(X) * 0.1
        R = Xt.mm(torch.inverse(P)).mm(Y)

        return R

    def get_pooled_feat(self, delta_p, prev_box, search_xyz, search_feat):

        # warp box
        new_box = self.warp_box(delta_p, prev_box)

        # crop feature inside box from 'search_feat', shape:[B, out_x, out_y, out_z, C]
        search_pooled_feat = self.roi_aware_pool(search_xyz, new_box, search_feat)
        search_pooled_feat_f = torch.flatten(search_pooled_feat, 1).unsqueeze(-1)  # [B, CxN, 1]
        return search_pooled_feat_f, new_box

    def warp_box(self, delta_p, prev_box):

        new_box = prev_box.clone()
        new_box[:, :, 0] = prev_box[:, :, 0] + delta_p[:, :, 0]
        new_box[:, :, 1] = prev_box[:, :, 1] + delta_p[:, :, 1]
        new_box[:, :, 2] = prev_box[:, :, 2] + delta_p[:, :, 2]
        new_box[:, :, 6] = prev_box[:, :, 6] + delta_p[:, :, -1]
        return new_box
