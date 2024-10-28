"""
Created by tye at 2021/11/10
"""

import torch
from torch import optim
import torch.nn.functional as F
from models import base_model
from models import p2b, bat, pvt, dsdm
from copy import deepcopy
import learn2learn as l2l
from datasets import points_utils
from utils.metrics import estimateAccuracy, estimateOverlap
from datasets.searchspace import KalmanFiltering
import numpy as np


class GST(base_model.BaseModel):

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        baseline = self.config.baseline
        self.baseline = globals()[baseline.lower()].__getattribute__(baseline.upper())(self.config)
        self.maml = l2l.algorithms.MAML(self.baseline,  # P2B: xcorr; BAT: rpn
                                        lr=self.config.meta_inner_lr,
                                        first_order=False)

    @torch.enable_grad()
    def meta_learn(self, support_batch, query_batch):
        support_batch_ = deepcopy(support_batch)
        learner = self.maml.clone()
        learner.train()
        for i in range(self.config.meta_inner_steps):
            loss, _ = self.run_model(support_batch, learner)
            learner.adapt(loss)

        query_batch = {key: torch.cat([support_batch_[key], query_batch[key]], 0) for key in support_batch_}
        loss, loss_dict = self.run_model(query_batch, learner)

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

    def run_model(self, batch, local_model):
        if self.config.baseline == "BAT":

            template = batch['template_points']
            search = batch['search_points']
            template_bc = batch['points2cc_dist_t']
            M = template.shape[1]
            N = search.shape[1]

            ## backbone
            # template_xyz, template_feature, sample_idxs_t = self.baseline.backbone(template, [M // 2, M // 4, M // 8])
            # search_xyz, search_feature, sample_idxs = self.baseline.backbone(search, [N // 2, N // 4, N // 8])
            # template_feature = self.baseline.conv_final(template_feature)
            # search_feature = self.baseline.conv_final(search_feature)

            ## prepare bc
            # pred_search_bc = self.baseline.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N // 8
            # pred_search_bc = pred_search_bc.transpose(1, 2)
            # sample_idxs_t = sample_idxs_t[:, :M // 8, None]
            # template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, self.config.bc_channel).long())

            ## box-aware xcorr
            # fusion_feature = self.baseline.xcorr(template_feature, search_feature,
            #                                      template_xyz, search_xyz,
            #                                      template_bc, pred_search_bc)

            ## proposal generation
            # estimation_boxes, estimation_cla, vote_xyz, center_xyzs = local_model(search_xyz, fusion_feature)

            # +++++++++++++++++++++++++++++++++++++meta learning for whole model ++++++++++++++++++++++++++++++++++++
            template_xyz, template_feature, sample_idxs_t = local_model.backbone(template, [M // 2, M // 4, M // 8])
            search_xyz, search_feature, sample_idxs = local_model.backbone(search, [N // 2, N // 4, N // 8])
            template_feature = local_model.conv_final(template_feature)
            search_feature = local_model.conv_final(search_feature)

            # prepare bc
            pred_search_bc = local_model.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N // 8
            pred_search_bc = pred_search_bc.transpose(1, 2)
            sample_idxs_t = sample_idxs_t[:, :M // 8, None]
            template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, self.config.bc_channel).long())

            # box-aware xcorr
            fusion_feature = local_model.xcorr(template_feature, search_feature,
                                               template_xyz, search_xyz,
                                               template_bc, pred_search_bc)

            estimation_boxes, estimation_cla, vote_xyz, center_xyzs = local_model.rpn(search_xyz, fusion_feature)

            end_points = {"estimation_boxes": estimation_boxes,
                          "vote_center": vote_xyz,
                          "pred_seg_score": estimation_cla,
                          "center_xyz": center_xyzs,
                          'sample_idxs': sample_idxs,
                          'estimation_cla': estimation_cla,
                          "vote_xyz": vote_xyz,
                          "pred_search_bc": pred_search_bc
                          }

        elif self.config.baseline == "P2B":
            template = batch['template_points']
            search = batch['search_points']
            M = template.shape[1]
            N = search.shape[1]

            # template_xyz, template_feature, _ = self.baseline.backbone(template, [M // 2, M // 4, M // 8])
            # search_xyz, search_feature, sample_idxs = self.baseline.backbone(search, [N // 2, N // 4, N // 8])
            # template_feature = self.baseline.conv_final(template_feature)
            # search_feature = self.baseline.conv_final(search_feature)
            #
            # fusion_feature = self.baseline.xcorr(template_feature, search_feature, template_xyz)
            # # fusion_feature = local_model(template_feature, search_feature, template_xyz)
            #
            # estimation_boxes, estimation_cla, vote_xyz, center_xyzs = local_model(search_xyz, fusion_feature)
            ## estimation_boxes, estimation_cla, vote_xyz, center_xyzs = self.baseline.rpn(search_xyz, fusion_feature)

            #+++++++++++++++++++++++++++++++meta learning for whole model +++++++++++++++++++++++++++++++++
            template_xyz, template_feature, _ = local_model.backbone(template, [M // 2, M // 4, M // 8])
            search_xyz, search_feature, sample_idxs = local_model.backbone(search, [N // 2, N // 4, N // 8])
            template_feature = local_model.conv_final(template_feature)
            search_feature = local_model.conv_final(search_feature)

            fusion_feature = local_model.xcorr(template_feature, search_feature, template_xyz)
            estimation_boxes, estimation_cla, vote_xyz, center_xyzs = local_model.rpn(search_xyz, fusion_feature)

            end_points = {"estimation_boxes": estimation_boxes,
                          "vote_center": vote_xyz,
                          "pred_seg_score": estimation_cla,
                          "center_xyz": center_xyzs,
                          'sample_idxs': sample_idxs,
                          'estimation_cla': estimation_cla,
                          "vote_xyz": vote_xyz,
                          }

        elif self.config.baseline == "PVT":
            template = batch['template_points']
            search = batch['search_points']
            template_bc = batch['points2cc_dist_t']
            t_voxel_coords, t_voxel_feats, t_voxel_inds \
                = batch['t_voxel_coords'], batch['t_voxel_feats'], batch['t_voxel_inds']
            s_voxel_coords, s_voxel_feats, s_voxel_inds \
                = batch['s_voxel_coords'], batch['s_voxel_feats'], batch['s_voxel_inds']

            # backbone, return # [B, 128, 3], [B, 128, N], [B, 128]
            template_xyz, template_feature, sample_idxs_t \
                = self.baseline.backbone(template, t_voxel_coords, t_voxel_feats, t_voxel_inds, self.baseline.num_sampled_point)
            search_xyz, search_feature, sample_idxs \
                = self.baseline.backbone(search, s_voxel_coords, s_voxel_feats, s_voxel_inds, self.baseline.num_sampled_point)

            # prepare bc
            pred_search_bc = self.baseline.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # [B, 9, 128]
            pred_search_bc = pred_search_bc.transpose(1, 2)
            template_bc = template_bc.gather(dim=1,
                                             index=sample_idxs_t.unsqueeze(-1).repeat(1, 1, template_bc.size(2)).long())

            # point-voxel based xcorr
            fusion_feature = self.baseline.xcorr(search_xyz, template_xyz, search_feature, template_feature)

            # proposal generation
            estimation_boxes, estimation_cla, vote_xyz, center_xyzs, \
            estimation_score, center_idxs = local_model(search_xyz, fusion_feature)

            search_center_idxs = torch.gather(sample_idxs.long(), 1, center_idxs.long())

            end_points = {"estimation_boxes": estimation_boxes,
                          "center_xyz": center_xyzs,
                          'sample_idxs': sample_idxs,
                          'estimation_cla': estimation_cla,
                          "vote_xyz": vote_xyz,
                          "pred_search_bc": pred_search_bc,
                          "estimation_score": estimation_score,
                          "search_center_idxs": search_center_idxs,
                          }
        else:
            return None

        # compute loss
        loss_dict = self.baseline.compute_loss(batch, end_points)
        loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
               + loss_dict['loss_box'] * self.config.box_weight \
               + loss_dict['loss_seg'] * self.config.seg_weight \
               + loss_dict['loss_vote'] * self.config.vote_weight \
               + loss_dict['loss_bc'] * self.config.bc_weight
        return loss, loss_dict

    def inner_loop(self, support_batch):
        local_model = deepcopy(self.baseline)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.config.meta_inner_lr)
        local_optim.zero_grad()

        # Optimize inner loop model on support set
        for i in range(self.config.meta_inner_steps):
            # Determine loss on the support set
            # loss = local_model.training_step(support_batch, i)
            loss, _ = self.run_model(support_batch, local_model)
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Reset gradients
            local_optim.zero_grad()

        return local_model

    @torch.enable_grad()
    def outer_loop(self, query_batch, local_model, mode="train"):
        self.train()
        opt = self.optimizers()
        opt.zero_grad()

        # Determine loss of query set
        # loss = local_model.training_step(query_batch, 0)
        loss, loss_dict = self.run_model(query_batch, local_model)

        # Calculate gradients for query set loss
        if mode == "train":
            loss.backward()

            # First-order approx. -> add gradients of finetuned and base model
            for p_global, p_local in zip(self.baseline.rpn.parameters(), local_model.parameters()):
                p_global.grad += p_local.grad

            opt.step()

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

    def training_step(self, batch, batch_idx):

        # split batch
        query_id = torch.randint(0, len(batch), [1])
        query_batch = batch[query_id]
        batch.pop(query_id)
        support_batch = {}
        for j, c in enumerate(batch):
            if j == 0:
                support_batch = c
            else:
                support_batch = {key: torch.cat([support_batch[key], c[key]], 0) for key in c.keys()}

        if self.current_epoch < self.config.warmup_num:
            support_batch = {key: torch.cat([support_batch[key], query_batch[key]], 0) for key in support_batch.keys()}
            output = self.baseline(support_batch)
            loss_dict = self.baseline.compute_loss(support_batch, output)

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
        else:
            # # Perform inner loop adaptation
            # local_model = self.inner_loop(support_batch)
            # # Perform outer loop optimization
            # self.outer_loop(query_batch, local_model, mode="train")
            # return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

            loss = self.meta_learn(support_batch, query_batch)
            return loss

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """

        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                if self.config.test_adapt:
                    template_pc, canonical_box = points_utils.cropAndCenterPC(sequence[0]['pc'], results_bbs[0],
                                                                              scale=self.config.model_bb_scale,
                                                                              offset=self.config.model_bb_offset)
                    template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size)
                    template_bc = points_utils.get_point_to_box_distance(template_points, canonical_box)

                    # gaussian = KalmanFiltering(bnd=[1, 1, 0.5, (5 if self.config.degrees else np.deg2rad(5))])
                    # sample_offsets = gaussian.sample(16)
                    sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=[8, 3])
                    sample_offsets[:, 2] = sample_offsets[:, 2] * (5 if self.config.degrees else np.deg2rad(5))

                    template_points_b, template_bc_b = [], []
                    search_points_b, seg_label_b, search_bbox_reg_b, search_bc_b = [], [], [], []
                    for sample_offset in sample_offsets:
                        sample_bb = points_utils.getOffsetBB(this_bb, sample_offset,
                                                             limit_box=self.config.limit_box, degrees=True,
                                                             use_z=self.config.use_z)
                        search_pc_crop = points_utils.generate_subwindow(sequence[0]['pc'], sample_bb,
                                                                         scale=self.config.search_bb_scale,
                                                                         offset=self.config.search_bb_offset)
                        search_box = points_utils.transform_box(this_bb, sample_bb)
                        seg_label = points_utils.get_in_box_mask(search_pc_crop, search_box).astype(int)
                        search_bbox_reg = np.array(
                            [search_box.center[0], search_box.center[1], search_box.center[2], -sample_offset[2]])

                        search_points, idx_s = points_utils.regularize_pc(search_pc_crop.points.T, self.config.search_size)
                        seg_label = seg_label[idx_s]
                        search_bc = points_utils.get_point_to_box_distance(search_points, search_box)

                        search_points_b.append(search_points)
                        seg_label_b.append(seg_label)
                        search_bbox_reg_b.append(search_bbox_reg)
                        search_bc_b.append(search_bc)

                        template_points_b.append(template_points)
                        template_bc_b.append(template_bc)

                    data_dict = {
                        'template_points': torch.stack(template_points_b).cuda().to(torch.float32),
                        'search_points': torch.stack(search_points_b).cuda().to(torch.float32),
                        'box_label': torch.stack(search_bbox_reg_b).cuda().to(torch.float32),
                        'seg_label': torch.stack(seg_label_b).cuda().to(torch.float32),
                        'points2cc_dist_t': torch.stack(template_bc_b).cuda().to(torch.float32),
                        'points2cc_dist_s': torch.stack(search_bc_b).cuda().to(torch.float32), }

                    self.maml.train()
                    local_model = self.maml.clone()
                    local_model.train()
                    for i in range(self.config.meta_inner_steps):
                        with torch.enable_grad():
                            loss, _ = self.run_model(data_dict, local_model)
                            # [print(p.names) for p in self.maml.parameters() if p.requires_grad]
                            local_model.adapt(loss, allow_nograd=True)
                    self.maml = local_model
            else:
                self.eval()
                # preparing search area
                search_pc_crop, ref_bb = self.baseline.generate_search_area(sequence, frame_id, results_bbs)

                # update template
                template_pc, canonical_box = self.baseline.generate_template(sequence, frame_id, results_bbs)

                # construct input dict
                data_dict = self.baseline.prepare_input(template_pc, search_pc_crop, canonical_box)

                end_points = self.maml.forward(data_dict)
                estimation_box = end_points['estimation_boxes']
                estimation_boxes_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                best_box_idx = estimation_boxes_cpu[:, 4].argmax()
                estimation_box_cpu = estimation_boxes_cpu[best_box_idx, 0:4]
                candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu, degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)
                results_bbs.append(candidate_box)
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances

        # return self.baseline.evaluate_one_sequence(sequence)

    # def evaluate_one_sequence(self, sequence):
    #     return self.baseline.evaluate_one_sequence(sequence)