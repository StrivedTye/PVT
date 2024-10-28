""" 
Created by tye at 2021/11/10
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.backbone.pointnet_pp import Pointnet_Backbone
from models.head.xcorr import P2B_XCorr
from models.head.rpn import P2BVoteNetRPN
from models import base_model
from models.head.refine import SequentialDescent
from pointnet2.utils import pytorch_utils as pt_utils
from datasets import points_utils
from datasets.searchspace import KalmanFiltering, ExhaustiveSearch
from utils.metrics import estimateOverlap, estimateAccuracy
from utils.loss_utils import SigmoidFocalClassificationLoss


class DSDM(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        self.backbone = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=True)
        self.conv_final = nn.Conv1d(256, self.config.feature_channel, kernel_size=1)

        self.mlp_bc = (pt_utils.Seq(3 + self.config.feature_channel)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.bc_channel, activation=None))

        self.xcorr = P2B_XCorr(feature_channel=self.config.feature_channel,
                               hidden_channel=self.config.hidden_channel,
                               out_channel=self.config.out_channel)

        self.rpn = P2BVoteNetRPN(self.config.feature_channel,
                                 vote_channel=self.config.vote_channel,
                                 num_proposal=self.config.num_proposal,
                                 normalize_xyz=self.config.normalize_xyz)

        self.sdm = SequentialDescent(num_voxels_roi=self.config.num_voxels_roi,
                                     max_pts_each_voxel=self.config.max_pts_each_voxel,
                                     learn_R=self.config.learn_R,
                                     num_R=self.config.num_R)
        self.focal_loss = SigmoidFocalClassificationLoss()

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
        search = input_dict['search_points']  # B, N, 3
        template_bc = input_dict['points2cc_dist_t']

        template_box = input_dict['box_tmpl'].unsqueeze(1)  # B, 1, 7
        search_box = input_dict['box_search']
        if search_box is not None:
            search_box = search_box.unsqueeze(1)
        search_box_input = input_dict['parallel_seeds']  # [B, 4, 7] only for closed-form solution

        # extract feature by pointNet++
        M, N = template.shape[1], search.shape[1]
        template_list_xyz, template_list_feat, sample_idxs_t = self.backbone(template, [M // 2, M // 4, M // 8])
        search_list_xyz, search_list_feat, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])

        template_xyz, search_xyz = template_list_xyz[0], search_list_xyz[0]
        template_feat, search_feat = template_list_feat[0], search_list_feat[0]
        template_mid_xyz, search_mid_xyz = template_list_xyz[-1], search_list_xyz[-1]
        template_mid_feat = self.conv_final(template_list_feat[-1])
        search_mid_feat = self.conv_final(search_list_feat[-1])

        # box cloud
        pred_search_bc = self.mlp_bc(torch.cat([search_mid_xyz.transpose(1, 2), search_mid_feat], dim=1))  # [B, 9, 128]
        pred_search_bc = pred_search_bc.transpose(1, 2)
        template_bc = template_bc.gather(dim=1, index=sample_idxs_t.unsqueeze(-1).repeat(1, 1, template_bc.size(2)).long())

        # correlation
        fusion_feature = self.xcorr(template_mid_feat, search_mid_feat, template_mid_xyz)

        # bbox candidates
        proposals, estimation_cla, vote_xyz, proposals_center = self.rpn(search_mid_xyz, fusion_feature)
        # only for using gaussian seeds
        # proposals = search_box_input[:,:,[0, 1, 2, 6]]
        # proposals_center = search_box_input[:,:,[0, 1, 2]]

        # sequential descent method
        search_box_all, delta_p_all, score_all = self.sdm(template_xyz, search_xyz,
                                                          template_feat, search_feat,
                                                          proposals, search_box_input,
                                                          template_box, search_box,
                                                          self.training)
        end_points = {"seed_boxes": proposals,
                      "center_xyz": proposals_center,
                      'sample_idxs': sample_idxs,
                      'estimation_cla': estimation_cla,
                      "vote_xyz": vote_xyz,
                      "pred_search_bc": pred_search_bc,
                      "search_box_all": search_box_all,
                      "delta_p_all": delta_p_all,
                      "score_all": score_all
                      }
        return end_points

    def compute_loss(self, input_dict, output):
        seg_label = input_dict['seg_label']  # [B, N]
        box_search = input_dict['box_search']  # [B, 7]
        box_tmpl = input_dict['box_tmpl']
        search_bc = input_dict['points2cc_dist_s']

        pred_search_bc = output['pred_search_bc']
        center = output['center_xyz']
        pred_box = output['seed_boxes']
        pred_seg = output['estimation_cla']  # [B, N]
        vote_xyz = output['vote_xyz']  # [B, N, 3]
        sample_idxs = output['sample_idxs']  # [B,N]
        search_box_all = output['search_box_all']  # [B, m*(num_R+1), 7]
        delta_p_all = output['delta_p_all']  # [B, m*num_R, 4]
        score_all = output['score_all']  # [B, m*(num_R+1), 1]

        num_seeds = pred_box.size(1)
        # gather the corresponding labels indicating foreground
        N = pred_seg.size(1)
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B, N
        search_bc = search_bc.gather(dim=1, index=sample_idxs[:, :N, None].repeat(1, 1, search_bc.size(-1)).long())

        # generate ground-truth delta_p for each step
        search_gt_state = box_search.unsqueeze(1).repeat(1, search_box_all.size(1), 1)  # [B, m*4, 7]
        step_delta_label = search_gt_state - search_box_all  # [B, m*4, 7]
        step_delta_label = step_delta_label[:, 0:-num_seeds, [0, 1, 2, 6]]  # [B, m*3, 3]

        # filter out proposals that is far away from gt
        dist = torch.sum((center - box_search[:, None, 0:3]) ** 2, dim=-1)
        dist = torch.sqrt(dist + 1e-6)
        objectness_label = torch.zeros_like(dist, dtype=torch.float32).cuda()
        objectness_mask = torch.zeros_like(dist, dtype=torch.float32).cuda()

        objectness_label[dist < 0.3] = 1
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        objectness_label_step = objectness_label.repeat(1, self.config.num_R+1)
        objectness_mask_step = objectness_mask.repeat(1, self.config.num_R+1)

        # seed: semantic
        # loss_seg = F.binary_cross_entropy_with_logits(pred_seg, seg_label)
        loss_seg = self.focal_loss(pred_seg.unsqueeze(-1),
                                   seg_label.unsqueeze(-1),
                                   torch.ones_like(seg_label))
        loss_seg = loss_seg.mean()

        # seed: vote
        loss_vote = F.smooth_l1_loss(vote_xyz, box_search[:, None, :3].expand_as(vote_xyz), reduction='none')
        loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        # seed: box
        loss_box = F.smooth_l1_loss(pred_box[:, :, :4],
                                    box_search[:, None, [0, 1, 2, 6]].expand_as(pred_box[:, :, :4]),
                                    reduction='none')
        loss_box = (loss_box.mean(2) * objectness_label).sum() / (objectness_label.sum() + 1e-6)

        # score
        loss_objective = F.binary_cross_entropy_with_logits(score_all.squeeze(-1), objectness_label_step,
                                                            pos_weight=torch.tensor([2.0], device=self.device))
        loss_objective = (loss_objective * objectness_mask_step).sum() / (objectness_mask_step.sum() + 1e-6)

        # progressive box loss
        loss_delta = F.smooth_l1_loss(delta_p_all, step_delta_label, reduction='none')
        loss_delta = (loss_delta.mean(2) * objectness_label_step[:, 0:-num_seeds]).sum() \
            / (objectness_label_step[:, 0:-num_seeds].sum() + 1e-6)

        # box cloud loss
        loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
        loss_bc = (loss_bc.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-6)

        out_dict = {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
            "loss_delta": loss_delta,
            "loss_bc": loss_bc
        }
        return out_dict

    def training_step(self, batch, batch_idx):
        self.train()
        end_points = self(batch)

        # compute loss
        loss_dict = self.compute_loss(batch, end_points)
        loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
               + loss_dict['loss_delta'] * self.config.delta_weight \
               + loss_dict['loss_box'] * self.config.seed_weight \
               + loss_dict['loss_vote'] * self.config.seed_weight \
               + loss_dict['loss_seg'] * self.config.seed_weight \
               + loss_dict['loss_bc'] * self.config.bc_weight

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_delta/train', loss_dict['loss_delta'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)

        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    'loss_vote': loss_dict['loss_vote'].item(),
                                                    'loss_objective': loss_dict['loss_objective'].item(),
                                                    'loss_delta': loss_dict['loss_delta'].item()},
                                           global_step=self.global_step)

        return loss

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):

        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T,
                                                            self.config.template_size,
                                                            seed=1)  # here "seed" is for random sample in numpy
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T,
                                                          self.config.search_size,
                                                          seed=1)

        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
        template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)

        box_tmpl = [0, 0, 0, template_box.wlh[1], template_box.wlh[0], template_box.wlh[2], 0]
        box_tmpl = torch.tensor(box_tmpl, device=self.device, dtype=torch.float32).reshape(1, 7)

        # prepare seeds
        num_seeds = 63
        search_space_sampler = kwargs.get('sampler', None)
        if search_space_sampler is not None:
            search_space_np = search_space_sampler.sample(num_seeds)
        else:
            search_space_sampler = ExhaustiveSearch(search_space=[[-1.0, 1.0],
                                                                  [-1.0, 1.0],
                                                                  [-10.0, 10.0]])
            search_space_np = search_space_sampler.sample(num_seeds)
        search_space = torch.from_numpy(search_space_np).to(self.device, torch.float32)

        parallel_seeds = box_tmpl.repeat(num_seeds+1, 1)  # [63+1, 7]
        parallel_seeds[1:, [0, 1]] += search_space[:, 0:2]
        parallel_seeds[1:, 6] += search_space[:, -1] * 3.1415926 / 180  # use radian

        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
            'points2cc_dist_t': template_bc_torch[None, ...],
            'box_tmpl': box_tmpl,
            'box_search': None,  # don't pass it when testing
            'parallel_seeds': parallel_seeds[None, ...],
            'search_space': search_space_np
        }
        return data_dict

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        search_space_sampler = KalmanFiltering([0.5, 0.5, 10])
        self.eval()
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                if not self.config.learn_R:

                    # preparing search area
                    search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)

                    # update template
                    template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)

                    # construct input dict
                    data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)

                    # forward
                    end_points = self(data_dict)

            else:

                # preparing search area
                search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)

                # update template
                template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)

                # construct input dict
                data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box, sampler=search_space_sampler)

                # forward
                end_points = self(data_dict)

                num_seeds = end_points["seed_boxes"].size(1)
                score_all = end_points["score_all"].sigmoid()  # [1, m*4, 1]
                search_box_all = end_points["search_box_all"]

                score_all_ = score_all.view(-1, num_seeds, 1)  # [4, m, 1]
                score_all_batch = torch.max(score_all_, dim=0)[0]  # [m, 1]
                score_all_batch = score_all_batch.detach().cpu().numpy()

                search_box_all = search_box_all.view(-1, 7)

                score_all = score_all.view(-1, 1)
                idx_box = torch.argmax(score_all, dim=0, keepdim=True)  # [1, 1]

                # **********************selection*****************
                selected_box = torch.gather(search_box_all, 0,
                                            idx_box.repeat(1, search_box_all.shape[-1]))
                selected_box_cpu = selected_box.detach().cpu().numpy()  # [1, 7]
                selected_box_cpu = selected_box_cpu[0, [0, 1, 2, 6]]

                candidate_box = points_utils.getOffsetBB(ref_bb, selected_box_cpu,
                                                         degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)
                results_bbs.append(candidate_box)
                search_space_sampler.addData(data_dict['search_space'], score_all_batch[1:, 0])

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances
