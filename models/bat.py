""" 
bat.py
Created by zenn at 2021/7/21 14:16
"""

import torch
from torch import nn
from models.backbone.pointnet import Pointnet_Backbone
from models.head.xcorr import BoxAwareXCorr, P2B_XCorr, PointVoxelXCorr
from models.head.rpn import P2BVoteNetRPN
from models import base_model
import torch.nn.functional as F
from datasets import points_utils
from pointnet2.utils import pytorch_utils as pt_utils
from copy import deepcopy


class BAT(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        if self.config.optimizer == 'sam':
            self.automatic_optimization = False
        self.backbone = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=False)
        self.conv_final = nn.Conv1d(256, self.config.feature_channel, kernel_size=1)
        self.mlp_bc = (pt_utils.Seq(3 + self.config.feature_channel)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.bc_channel, activation=None))

        self.xcorr = BoxAwareXCorr(feature_channel=self.config.feature_channel,
                                   hidden_channel=self.config.hidden_channel,
                                   out_channel=self.config.out_channel,
                                   k=self.config.k,
                                   use_search_bc=self.config.use_search_bc,
                                   use_search_feature=self.config.use_search_feature,
                                   bc_channel=self.config.bc_channel)
        # self.xcorr_fs = P2B_XCorr(feature_channel=self.config.feature_channel,
        #                           hidden_channel=self.config.hidden_channel,
        #                           out_channel=self.config.out_channel)
        # self.mlp_switch = (pt_utils.Seq(self.config.feature_channel)
        #                    .conv1d(self.config.feature_channel, bn=True)
        #                    .conv1d(2, activation=None))

        # self.xcorr_vs = PointVoxelXCorr(num_levels=self.config.num_levels,
        #                                 base_scale=self.config.base_scale,
        #                                 truncate_k=self.config.truncate_k,
        #                                 num_knn=self.config.num_knn,
        #                                 feat_channel=self.config.out_channel // 2)

        self.rpn = P2BVoteNetRPN(self.config.feature_channel,
                                 vote_channel=self.config.vote_channel,
                                 num_proposal=self.config.num_proposal,
                                 normalize_xyz=self.config.normalize_xyz)

    def prepare_input(self, template_pc, search_pc, template_box):
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
        template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
            'points2cc_dist_t': template_bc_torch[None, ...]
        }
        return data_dict

    def compute_loss(self, data, output):
        search_bc = data['points2cc_dist_s']
        estimation_cla = output['estimation_cla']  # B,N
        N = estimation_cla.shape[1]
        seg_label = data['seg_label']
        sample_idxs = output['sample_idxs']  # B,N
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        search_bc = search_bc.gather(dim=1, index=sample_idxs[:, :N, None].repeat(1, 1, self.config.bc_channel).long())
        # update label
        data['seg_label'] = seg_label
        data['points2cc_dist_s'] = search_bc

        out_dict = super(BAT, self).compute_loss(data, output)
        search_bc = data['points2cc_dist_s']
        pred_search_bc = output['pred_search_bc']
        seg_label = data['seg_label']
        loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
        loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)
        out_dict["loss_bc"] = loss_bc
        return out_dict

    def forward(self, input_dict):
        """
        :param input_dict:
        {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        'points2cc_dist_t': template_bc,
        'points2cc_dist_s': search_bc,
        }

        :return:
        """

        template = input_dict['template_points']
        search = input_dict['search_points']
        template_bc = input_dict['points2cc_dist_t']
        M = template.shape[1]
        N = search.shape[1]

        # backbone
        template_xyz, template_feature, sample_idxs_t = self.backbone(template, [M // 2, M // 4, M // 8])
        search_xyz, search_feature, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])
        template_feature = self.conv_final(template_feature)
        search_feature = self.conv_final(search_feature)

        # prepare bc
        pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N // 8
        pred_search_bc = pred_search_bc.transpose(1, 2)
        sample_idxs_t = sample_idxs_t[:, :M // 8, None]
        template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, self.config.bc_channel).long())

        # box-aware xcorr
        fusion_feature = self.xcorr(template_feature, search_feature,
                                    template_xyz, search_xyz,
                                    template_bc, pred_search_bc)
        # fusion_feature = self.xcorr_vs(search_xyz, template_xyz,
        #                                search_feature, template_feature,
        #                                pred_search_bc, template_bc)
        # fusion_feature = torch.cat([fusion_feature_bc, fusion_feature_vs], dim=1)

        # fusion_feature_bc = self.xcorr(template_feature, search_feature,
        #                                template_xyz, search_xyz,
        #                                template_bc, pred_search_bc)
        # fusion_feature_fs = self.xcorr_fs(template_feature, search_feature, template_xyz)
        # switch = self.mlp_switch(torch.avg_pool1d(search_feature, kernel_size=N//8))
        # switch = torch.softmax(switch, 1)  # B, 2, 1
        # switch = F.gumbel_softmax(switch, tau=0.5, hard=True)
        # fusion_feature = fusion_feature_fs * switch[:, 0:1, :] + fusion_feature_bc * switch[:, 1:2, :]

        # for decorrelation
        fusion_feature_avg = F.avg_pool1d(fusion_feature, N // 8).squeeze(2)  # B, C, 1

        # proposal generation
        estimation_boxes, estimation_cla, vote_xyz, center_xyzs = self.rpn(search_xyz, fusion_feature)
        end_points = {"estimation_boxes": estimation_boxes,
                      "vote_center": vote_xyz,
                      "pred_seg_score": estimation_cla,
                      "center_xyz": center_xyzs,
                      'sample_idxs': sample_idxs,
                      'estimation_cla': estimation_cla,
                      "vote_xyz": vote_xyz,
                      "pred_search_bc": pred_search_bc,
                      "fusion_avg": fusion_feature_avg,
                      "fusion_feat": fusion_feature
                      }
        return end_points

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        """
        {"estimation_boxes": estimation_boxs.transpose(1, 2).contiguous(),
                  "vote_center": vote_xyz,
                  "pred_seg_score": estimation_cla,
                  "center_xyz": center_xyzs,
                  "seed_idxs":
                  "seg_label"
                  "pred_search_bc": pred_search_bc
        }
        """

        # compute loss
        if self.config.optimizer == 'sam':
            optimizer = self.optimizers()

            # first forward-backward pass
            def _enable(module):
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) and hasattr(module, "backup_momentum"):
                    module.momentum = module.backup_momentum
            self.apply(_enable)

            end_points = self(batch)
            loss_dict = self.compute_loss(deepcopy(batch), end_points)
            loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
                   + loss_dict['loss_box'] * self.config.box_weight \
                   + loss_dict['loss_seg'] * self.config.seg_weight \
                   + loss_dict['loss_vote'] * self.config.vote_weight \
                   + loss_dict['loss_bc'] * self.config.bc_weight

            self.manual_backward(loss, optimizer)
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            def _disable(module):
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.backup_momentum = module.momentum
                    module.momentum = 0
            self.apply(_disable)

            end_points = self(batch)
            loss_dict = self.compute_loss(batch, end_points)
            loss_2 = loss_dict['loss_objective'] * self.config.objectiveness_weight \
                     + loss_dict['loss_box'] * self.config.box_weight \
                     + loss_dict['loss_seg'] * self.config.seg_weight \
                     + loss_dict['loss_vote'] * self.config.vote_weight \
                     + loss_dict['loss_bc'] * self.config.bc_weight

            self.manual_backward(loss_2, optimizer)
            optimizer.second_step(zero_grad=True)
        else:
            end_points = self(batch)
            loss_dict = self.compute_loss(batch, end_points)
            loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
                   + loss_dict['loss_box'] * self.config.box_weight \
                   + loss_dict['loss_seg'] * self.config.seg_weight \
                   + loss_dict['loss_vote'] * self.config.vote_weight \
                   + loss_dict['loss_bc'] * self.config.bc_weight

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)

        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    'loss_vote': loss_dict['loss_vote'].item(),
                                                    'loss_objective': loss_dict['loss_objective'].item(),
                                                    'loss_bc': loss_dict['loss_bc'].item()},
                                           global_step=self.global_step)

        return loss