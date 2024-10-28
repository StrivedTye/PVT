""" 
p2b.py
Created by zenn at 2021/5/9 13:47
"""

from torch import nn
import torch.nn.functional as F
from models.backbone.pointnet import Pointnet_Backbone
from models.head.xcorr import P2B_XCorr
from models.head.rpn import P2BVoteNetRPN
from models import base_model
from copy import deepcopy


class P2B(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        if self.config.optimizer == 'sam':
            self.automatic_optimization = False
        self.backbone = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=False)
        self.conv_final = nn.Conv1d(256, self.config.feature_channel, kernel_size=1)

        self.xcorr = P2B_XCorr(feature_channel=self.config.feature_channel,
                               hidden_channel=self.config.hidden_channel,
                               out_channel=self.config.out_channel)
        self.rpn = P2BVoteNetRPN(self.config.feature_channel,
                                 vote_channel=self.config.vote_channel,
                                 num_proposal=self.config.num_proposal,
                                 normalize_xyz=self.config.normalize_xyz)

    def forward(self, input_dict):
        """
        :param input_dict:
        {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        }

        :return:
        """
        template = input_dict['template_points']
        search = input_dict['search_points']
        M = template.shape[1]
        N = search.shape[1]

        template_xyz, template_feature, _ = self.backbone(template, [M // 2, M // 4, M // 8])
        search_xyz, search_feature, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])
        template_feature = self.conv_final(template_feature)
        search_feature = self.conv_final(search_feature)
        fusion_feature = self.xcorr(template_feature, search_feature, template_xyz)

        # for decorrelation
        fusion_feature_avg = F.avg_pool1d(fusion_feature, N // 8).squeeze(2)  # B, C, 1

        estimation_boxes, estimation_cla, vote_xyz, center_xyzs = self.rpn(search_xyz, fusion_feature)
        end_points = {"estimation_boxes": estimation_boxes,
                      "vote_center": vote_xyz,
                      "pred_seg_score": estimation_cla,
                      "center_xyz": center_xyzs,
                      'sample_idxs': sample_idxs,
                      'estimation_cla': estimation_cla,
                      "vote_xyz": vote_xyz,
                      "fusion_avg": fusion_feature_avg,
                      "fusion_feat": fusion_feature
                      }
        return end_points

    def compute_loss(self, data, output):
        estimation_cla = output['estimation_cla']  # B,N
        N = estimation_cla.shape[1]
        seg_label = data['seg_label']
        sample_idxs = output['sample_idxs']  # B,N
        # update label
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        data["seg_label"] = seg_label
        out_dict = super(P2B, self).compute_loss(data, output)
        out_dict['loss_bc'] = out_dict['loss_objective']
        return out_dict

    def training_step(self, batch, batch_idx):
        """
        {"estimation_boxes": estimation_boxs.transpose(1, 2).contiguous(),
                  "vote_center": vote_xyz,
                  "pred_seg_score": estimation_cla,
                  "center_xyz": center_xyzs,
                  "seed_idxs":
                  "seg_label"
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
                   + loss_dict['loss_vote'] * self.config.vote_weight

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
                     + loss_dict['loss_vote'] * self.config.vote_weight

            self.manual_backward(loss_2, optimizer)
            optimizer.second_step(zero_grad=True)
        else:
            end_points = self(batch)
            loss_dict = self.compute_loss(batch, end_points)
            loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
                   + loss_dict['loss_box'] * self.config.box_weight \
                   + loss_dict['loss_seg'] * self.config.seg_weight \
                   + loss_dict['loss_vote'] * self.config.vote_weight

        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    'loss_vote': loss_dict['loss_vote'].item(),
                                                    'loss_objective': loss_dict['loss_objective'].item()},
                                           global_step=self.global_step)

        return loss


