""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
"""
import torch
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision, TorchAverage
from utils.metrics import estimateOverlap, estimateAccuracy
import torch.nn.functional as F

from utils.reweighting import weight_learner
from utils.sam import SAM

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import time

import yaml


class BaseModel(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.hd = TorchAverage()
        self.cd = TorchAverage()

        self.times_tracklet = []
        self.tracklet_count = 0

        if getattr(self.config, "re_weight", False):
            # self.pre_features = torch.zeros(self.config.batch_size, self.config.feature_channel)
            # self.pre_weight = torch.ones(self.config.batch_size, 1)
            self.register_buffer('pre_features', torch.zeros(self.config.batch_size, self.config.feature_channel), persistent=False)
            self.register_buffer('pre_weight', torch.ones(self.config.batch_size, 1), persistent=False)

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        elif self.config.optimizer.lower() == 'sam':
            optimizer = SAM(self.parameters(), torch.optim.SGD, lr=self.config.lr, momentum=0.9)
        else:
            betas = getattr(self.config, 'betas', (0.5, 0.999))
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=tuple(betas), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        """

        :param data: input data
        :param output:
        :return:
        """
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        estimation_cla = output['estimation_cla']  # B,N
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]
        seg_label = data['seg_label']
        box_label = data['box_label']  # B,4

        loss_seg = F.binary_cross_entropy_with_logits(estimation_cla, seg_label, reduction='none')  # B, N

        loss_vote = F.smooth_l1_loss(vote_xyz, box_label[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
        # loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float32)
        objectness_label[dist < 0.3] = 1

        objectness_score = estimation_boxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float32)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1

        loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                            pos_weight=torch.tensor([2.0], device=self.device))  #B, K
        # loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

        loss_box = F.smooth_l1_loss(estimation_boxes[:, :, :4],
                                    box_label[:, None, :4].expand_as(estimation_boxes[:, :, :4]),
                                    reduction='none')  # B, K, 4
        # loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        if getattr(self.config, "re_weight", False):
            cur_features = output["fusion_avg"]
            pre_features, pre_weight = self.pre_features, self.pre_weight

            if self.current_epoch >= self.config.rw_warmup:
                weight, pre_features, pre_weight = weight_learner(cur_features, pre_features, pre_weight,
                                                                  self.config, self.current_epoch, self.global_step)
            else:
                weight = 1 / cur_features.size(0) * torch.ones(cur_features.size(0), 1).cuda()
            self.pre_features.data.copy_(pre_features)
            self.pre_weight.data.copy_(pre_weight)

            loss_seg = loss_seg.mean(1).view(1, -1).mm(weight)

            loss_vote = (loss_vote.mean(2) * seg_label).sum(1) / (seg_label.sum(1) + 1e-6)
            loss_vote = loss_vote.view(1, -1).mm(weight)

            loss_objective = (loss_objective * objectness_mask).sum(1) / (objectness_mask.sum(1) + 1e-6)
            loss_objective = loss_objective.view(1, -1).mm(weight)

            loss_box = (loss_box.mean(2) * objectness_label).sum(1) / (objectness_label.sum(1) + 1e-6)
            loss_box = loss_box.view(1, -1).mm(weight)
        else:
            loss_seg = loss_seg.mean()
            loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)
            loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
            loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        loss_dict = {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
        }

        return loss_dict

    def generate_template(self, sequence, current_frame_id, results_bbs):
        """
        generate template for evaluating.
        the template can be updated using the previous predictions.
        :param sequence: the list of the whole sequence
        :param current_frame_id:
        :param results_bbs: predicted box for previous frames
        :return:
        """
        first_pc = sequence[0]['pc']
        previous_pc = sequence[current_frame_id - 1]['pc']
        if "firstandprevious".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([first_pc, previous_pc],
                                                               [results_bbs[0], results_bbs[current_frame_id - 1]],
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        elif "first".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(first_pc, results_bbs[0],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "previous".upper() in self.config.hape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(previous_pc, results_bbs[current_frame_id - 1],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "all".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([frame["pc"] for frame in sequence[:current_frame_id]],
                                                               results_bbs,
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        return template_pc, canonical_box

    def generate_search_area(self, sequence, current_frame_id, results_bbs):
        """
        generate search area for evaluating.

        :param sequence:
        :param current_frame_id:
        :param results_bbs:
        :return:
        """
        this_bb = sequence[current_frame_id]["3d_bbox"]
        this_pc = sequence[current_frame_id]["pc"]
        if "previous_result".upper() in self.config.reference_BB.upper():
            ref_bb = results_bbs[-1]
        elif "previous_gt".upper() in self.config.reference_BB.upper():
            previous_bb = sequence[current_frame_id - 1]["3d_bbox"]
            ref_bb = previous_bb
        elif "current_gt".upper() in self.config.reference_BB.upper():
            ref_bb = this_bb
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset)
        return search_pc_crop, ref_bb

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
        """
        construct input dict for evaluating
        :param template_pc:
        :param search_pc:
        :param template_box:
        :return:
        """
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32, requires_grad=True)
        template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
        template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
            'points2cc_dist_t': template_bc_torch[None, ...]
        }
        return data_dict

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        cd_imperception = []
        hd_imperception = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # preparing search area
                search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)

                # update template
                template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)

                # construct input dict
                data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)

                end_points = self(data_dict)
                estimation_box = end_points['estimation_boxes']
                estimation_boxes_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                best_box_idx = estimation_boxes_cpu[:, 4].argmax()
                estimation_box_cpu = estimation_boxes_cpu[best_box_idx, 0:4]
                candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu,
                                                         degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)
                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)

        # for saving testing boxes.
        save_test_path = getattr(self.config, 'save_test_path', None)
        if save_test_path is not None and self.config.test:
            save_test_path = os.path.join(self.config.save_test_path, self.config.category_name)
            os.makedirs(save_test_path, exist_ok=True)
            file_name = os.path.join(save_test_path, f'{self.tracklet_count:04.0f}.txt')
            results_bbs_saved = [b.corners().flatten() for b in results_bbs]
            np.savetxt(file_name, results_bbs_saved, fmt='%f')

        return ious, distances, hd_imperception, cd_imperception

    def validation_step(self, batch, batch_idx):
        # if batch_idx == 4:  # for test visual
        if batch_idx != -1:

            sequence = batch[0]  # unwrap the batch with batch size = 1

            ious, distances, hd, cd = self.evaluate_one_sequence(sequence)
            # exit()

            # update metrics
            self.success(torch.tensor(ious, device=self.device))
            self.prec(torch.tensor(distances, device=self.device))

            self.log('success/test', self.success, on_step=True, on_epoch=True)
            self.log('precision/test', self.prec, on_step=True, on_epoch=True)
        else:
            print("skip: ", batch_idx)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

    def test_step(self, batch, batch_idx):

        # if batch_idx >= 97:  # for test visual
        if batch_idx != -1:

            sequence = batch[0]  # unwrap the batch with batch size = 1

            start_time = time.time()
            ious, distances, hd, cd = self.evaluate_one_sequence(sequence)
            end_start = time.time()

            self.times_tracklet.append((end_start - start_time)/len(sequence))
            # exit()

            self.tracklet_count += 1

            # update metrics
            self.success(torch.tensor(ious, device=self.device))
            self.prec(torch.tensor(distances, device=self.device))

            self.log('success/test', self.success, on_step=True, on_epoch=True)
            self.log('precision/test', self.prec, on_step=True, on_epoch=True)

        # else:
        #     print("skip: ", batch_idx)

    def test_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

        if self.config.test:
            fps = 1 / (sum(self.times_tracklet) / len(self.times_tracklet))
            print('Running frame per second: {:04.2f}'.format(fps))
            print('HD:{:f}, CD: {:f}'.format(self.hd.compute(), self.cd.compute()))