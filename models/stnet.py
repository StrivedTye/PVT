"""
STNet.py
Created by tye at 2022/12/27
"""
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

from models.backbone.pointnet2_trans import Pointnet_Backbone
from models.head.xcorr_trans import STNetXCorr
from models.head.rpn import VoxelRPN
from models.head.xcorr import BoxAwareXCorr, P2B_XCorr, PointVoxelXCorr
from models import base_model

from datasets import points_utils
from utils.loss_utils import FocalLoss, RegL1Loss, _sigmoid
from utils.box_coder import mot_decode
from utils.metrics import estimateOverlap, estimateAccuracy
from pointnet2.utils import pytorch_utils as pt_utils

from utils.reweighting import weight_learner
from utils.sam import SAM

from hausdorff import hausdorff_distance
from chamferdist import ChamferDistance


class STNET(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        if self.config.optimizer == 'sam':
            self.automatic_optimization = False

        # ---------------------------- hyper-parameters --------------------------
        config = self.cal_params_by_config(config)
        self.config = config

        self.voxel_size = torch.from_numpy(config.voxel_size).float()
        self.voxel_area = config.voxel_area
        self.scene_ground = torch.from_numpy(config.scene_ground).float()
        self.min_img_coord = torch.from_numpy(config.min_img_coord).float()
        self.xy_size = torch.from_numpy(config.xy_size).float()

        self.mode = not config.test
        self.feat_emb = config.feat_emb
        self.iters = config.iters
        self.attention_type = 'linear'
        self.knn_num = config.knn_num

        self.backbone_net = Pointnet_Backbone(0, use_xyz=config.use_xyz)

        # -------------------------- xcorr Attention ------------------------
        self.xcorr = STNetXCorr(self.feat_emb, self.iters, self.attention_type, self.knn_num)
        self.seg = (pt_utils.Seq(32).conv1d(32, bn=True).conv1d(32, bn=True).conv1d(1, activation=None))

        # self.xcorr = BoxAwareXCorr(feature_channel=32,
        #                            hidden_channel=64,
        #                            out_channel=32,
        #                            k=4,
        #                            use_search_bc=False,
        #                            use_search_feature=False,
        #                            bc_channel=9)

        self.mlp_bc = (pt_utils.Seq(3 + 32)
                       .conv1d(64, bn=True)
                       .conv1d(64, bn=True)
                       .conv1d(9, activation=None))

        # -------------------------- RPN detection ------------------------
        self.rpn = VoxelRPN(self.voxel_area, self.scene_ground, self.mode, self.voxel_size, feat_dim=3+32)

        self.focal_loss = FocalLoss().cuda()
        self.l1_loss_loc = RegL1Loss().cuda()
        self.l1_loss_z = RegL1Loss().cuda()

    def compute_loss(self, data, output):
        output['pred_hm'] = _sigmoid(output['pred_hm'])

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

            loss_reg_hm = self.focal_loss(output['pred_hm'], data['hot_map'], reduction="none") # [b]
            loss_reg_hm = loss_reg_hm.view(1, -1).mm(weight)

            loss_reg_loc = self.l1_loss_loc(output['pred_loc'], data['index_offsets'], data['local_offsets'], reduction="none")
            loss_reg_loc = loss_reg_loc.mean(2, 1).view(1, -1).mm(weight)

            loss_reg_z = self.l1_loss_z(output['pred_z_axis'], data['index_center'], data['z_axis'], reduction="none")
            loss_reg_z = loss_reg_z.mean(2, 1).view(1, -1).mm(weight)

            seg_label = data['seg_label']
            loss_seg = F.binary_cross_entropy_with_logits(output['pred_seg'], seg_label, reduction="none")  # B, N
            loss_seg = loss_seg.mean(1).view(1, -1).mm(weight)

            loss_bc = F.smooth_l1_loss(output['pred_search_bc'], data['points2cc_dist_s'], reduction='none')
            loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)

        else:
            loss_reg_hm = self.focal_loss(output['pred_hm'], data['hot_map'])
            loss_reg_loc = self.l1_loss_loc(output['pred_loc'], data['index_offsets'], data['local_offsets'])
            loss_reg_z = self.l1_loss_z(output['pred_z_axis'], data['index_center'], data['z_axis'])

            seg_label = data['seg_label']
            loss_seg = F.binary_cross_entropy_with_logits(output['pred_seg'], seg_label)  # B, N

            loss_bc = F.smooth_l1_loss(output['pred_search_bc'], data['points2cc_dist_s'], reduction='none')
            loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)

        out_dict = {
            "loss_reg_hm": loss_reg_hm,
            "loss_reg_loc": loss_reg_loc,
            "loss_reg_z": loss_reg_z,
            "loss_seg": loss_seg,
            "loss_bc": loss_bc
        }

        return out_dict

    def forward(self, input_dict):

        template = input_dict['template_points']
        search = input_dict['search_points']
        template_bc = input_dict['points2cc_dist_t']

        M = template.shape[1]
        N = search.shape[1]

        # ---------------------- Siamese Network ----------------------
        template_xyz, template_feature = self.backbone_net(template, [256, 128, 64])
        search_xyz, search_feature = self.backbone_net(search, [512, 256, 128])

        # prepare bc
        pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N
        pred_search_bc = pred_search_bc.transpose(1, 2)

        # -------------------- correlation learning  ---------------
        fusion_feature = self.xcorr(search_feature, search_xyz, template_feature, template_xyz)
        # fusion_feature = self.xcorr(template_feature, search_feature,
        #                                template_xyz, search_xyz,
        #                                template_bc, pred_search_bc)

        fusion_feature_avg = F.avg_pool1d(fusion_feature, N).squeeze(2)  # B, C, 1

        pred_seg = self.seg(fusion_feature).squeeze(1)

        # ---------------------- Detection -------------------------
        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(),
                                        # pred_seg.sigmoid().unsqueeze(1),
                                        fusion_feature), dim=1)
        pred_hm, pred_loc, pred_z_axis = self.rpn(fusion_xyz_feature, search_xyz)

        end_points = {"pred_hm": pred_hm,
                      "pred_loc": pred_loc,
                      "pred_z_axis": pred_z_axis,
                      "fusion_avg": fusion_feature_avg,
                      "pred_seg": pred_seg,
                      "pred_search_bc": pred_search_bc,
                      }
        return end_points

    def training_step(self, batch, batch_idx, dataloader_idx=None):

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
            loss = loss_dict['loss_reg_hm'] * self.config.hm_weight \
                   + loss_dict['loss_reg_loc'] * self.config.loc_weight \
                   + loss_dict['loss_reg_z'] * self.config.z_weight \
                   + loss_dict['loss_seg'] * self.config.seg_weight \
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
            loss_2 = loss_dict['loss_reg_hm'] * self.config.hm_weight \
                     + loss_dict['loss_reg_loc'] * self.config.loc_weight \
                     + loss_dict['loss_reg_z'] * self.config.z_weight \
                     + loss_dict['loss_seg'] * self.config.seg_weight \
                     + loss_dict['loss_bc'] * self.config.bc_weight

            self.manual_backward(loss_2, optimizer)
            optimizer.second_step(zero_grad=True)
        else:
            end_points = self(batch)
            loss_dict = self.compute_loss(batch, end_points)
            loss = loss_dict['loss_reg_hm'] * self.config.hm_weight \
                   + loss_dict['loss_reg_loc'] * self.config.loc_weight \
                   + loss_dict['loss_reg_z'] * self.config.z_weight \
                   + loss_dict['loss_seg'] * self.config.seg_weight \
                   + loss_dict['loss_bc'] * self.config.bc_weight

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_hm/train', loss_dict['loss_reg_hm'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_loc/train', loss_dict['loss_reg_loc'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_zaxis/train', loss_dict['loss_reg_z'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_reg_hm': loss_dict['loss_reg_hm'].item(),
                                                    'loss_reg_loc': loss_dict['loss_reg_loc'].item(),
                                                    'loss_reg_z': loss_dict['loss_reg_z'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    'loss_bc': loss_dict['loss_bc'].item(),
                                                    },
                                           global_step=self.global_step)
        return loss

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
        area_extents = torch.tensor(self.config.area_extents).reshape(3, 2).numpy()
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset,
                                                         limit_area=area_extents)
        return search_pc_crop, ref_bb

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

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        cd_imperception = []
        hd_imperception = []

        self.eval()
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

                if getattr(self.config, 'attack', False):
                    adv_path = './adv_examples/stnet/' + 'ba3_' + self.config.category_name + '/'
                    os.makedirs(adv_path, exist_ok=True)
                    search_points_clean = data_dict['search_points'][0]
                    np.savetxt(adv_path+f'clean_{self.tracklet_count:04.0f}_{frame_id:04d}.txt',search_points_clean.detach().cpu().numpy(), fmt='%f')

                    # data_dict = self.attack_pgd(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attack_fgsm(data_dict, this_bb, results_bbs[-1])
                    #data_dict = self.attack_gauss(data_dict)
                    #data_dict = self.attack_cw(data_dict, this_bb, results_bbs[-1])

                    #data_dict = self.attack_trans(data_dict)
                    data_dict = self.attack_ba(data_dict, this_bb, results_bbs[-1], frame_id)
                    #data_dict = self.attacker.attack_fgsm(data_dict, this_bb, results_bbs[-1])
                    #data_dict = self.attacker.attack_pgd(data_dict, this_bb, results_bbs[-1])
                    #data_dict = self.attacker.attack_cw(data_dict, this_bb, results_bbs[-1])

                    search_points_adv = data_dict['search_points'][0]
                    np.savetxt(adv_path+f'adv_{self.tracklet_count:04.0f}_{frame_id:04d}.txt', search_points_adv.detach().cpu().numpy(), fmt='%f')

                    hd_imperception.append(hausdorff_distance(search_points_adv.detach().cpu().numpy(),
                                                              search_points_clean.detach().cpu().numpy(),
                                                              distance='euclidean'))
                    cd_imperception.append(self.chamferdist(search_points_clean.unsqueeze(0),
                                                            search_points_adv.unsqueeze(0),
                                                            bidirectional=True).detach().cpu().item())

                # forward
                try:
                    end_points = self(data_dict)

                    pred_hm = end_points["pred_hm"].sigmoid_()
                    pred_loc = end_points["pred_loc"]
                    pred_z_axis = end_points["pred_z_axis"]

                    xy_img_z_ry = mot_decode(pred_hm, pred_loc, pred_z_axis, K=1)

                    xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
                    xy_img_z_ry_cpu[:, :2] = (xy_img_z_ry_cpu[:, :2] + self.min_img_coord.numpy()) * self.xy_size.numpy()
                    estimate_box = xy_img_z_ry_cpu[0]
                    candidate_box = points_utils.getOffsetBB(ref_bb, estimate_box[:4],
                                                             degrees=self.config.degrees,
                                                             use_z=self.config.use_z,
                                                             limit_box=self.config.limit_box)

                    results_bbs.append(candidate_box)

                except RuntimeError:
                    results_bbs.append(results_bbs[-1])
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances, hd_imperception, cd_imperception

    def cal_params_by_config(self, config):

        voxel_size = np.array(config.voxel_size)
        xy_voxle_size = np.array(config.xy_size) * config.downsample

        area_extents = np.array(config.area_extents).reshape(3, 2)
        xy_area_extents = np.array(config.xy_area_extents).reshape(2, 2)
        voxel_extents_transpose = area_extents.transpose()
        extents_transpose = xy_area_extents.transpose()

        scene_ground = voxel_extents_transpose[0]
        voxel_grid_size = np.ceil(voxel_extents_transpose[1] / voxel_size) - np.floor(
            voxel_extents_transpose[0] / voxel_size)
        voxel_grid_size = voxel_grid_size.astype(np.int32)

        min_img_coord = np.floor(extents_transpose[0] / xy_voxle_size)
        max_img_coord = np.ceil(extents_transpose[1] / xy_voxle_size) - 1
        img_size = ((max_img_coord - min_img_coord) + 1).astype(np.int32)  # [w, h]

        config.voxel_size = voxel_size
        config.voxel_area = voxel_grid_size
        config.scene_ground = scene_ground
        config.min_img_coord = min_img_coord
        config.xy_size = xy_voxle_size
        return config

    def run_track(self, data_dict, ref_bb):
        # forward
        end_points = self(data_dict)

        pred_hm = end_points["pred_hm"].sigmoid_()
        pred_loc = end_points["pred_loc"]
        pred_z_axis = end_points["pred_z_axis"]

        xy_img_z_ry = mot_decode(pred_hm, pred_loc, pred_z_axis, K=1)

        xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
        xy_img_z_ry_cpu[:, :2] = (xy_img_z_ry_cpu[:, :2] + self.min_img_coord.numpy()) * self.xy_size.numpy()
        estimate_box = xy_img_z_ry_cpu[0]

        candidate_box = points_utils.getOffsetBB(ref_bb, estimate_box[:4],
                                                 degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)

        return candidate_box, end_points

    def attack_loss(self, end_point, gt_box, prev_box, target_attack=True):
        gt_box_normal = points_utils.transform_box(gt_box, prev_box)
        pred_hm = end_point['pred_hm']  # b, c, h, w
        cy_ind, cx_ind = round(pred_hm.size(2)/2), round(pred_hm.size(3)/2)

        if target_attack:
            cx_ind = cx_ind - gt_box_normal.center[0]  # move adversely
            cy_ind = cy_ind - gt_box_normal.center[1]
            cx_ind = np.clip(cx_ind, 0, pred_hm.size(3)-1).astype(np.int32)
            cy_ind = np.clip(cy_ind, 0, pred_hm.size(2)-1).astype(np.int32)
            adv_hm = torch.zeros_like(pred_hm)
            adv_hm[:, :, cy_ind, cx_ind] = 1
            loss = self.focal_loss(pred_hm, adv_hm)
            return -loss  # for gradien ascend
        else:
            cx_ind = cx_ind + gt_box_normal.center[0]
            cy_ind = cy_ind + gt_box_normal.center[1]
            cx_ind = np.clip(cx_ind, 0, pred_hm.size(3)-1).astype(np.int32)
            cy_ind = np.clip(cy_ind, 0, pred_hm.size(2)-1).astype(np.int32)
            gt_hm = torch.zeros_like(pred_hm)
            gt_hm[:, :, cy_ind, cx_ind] = 1
            # gt_hm[0, [cy_ind, cy_ind, cy_ind + 1, cy_ind - 1], [cx_ind - 1, cx_ind + 1, cx_ind, cx_ind]] = 0.8
            loss = self.focal_loss(pred_hm, gt_hm)
            return loss  # for gradien ascend

    def attack_fgsm(self, data_dict, gt_box, prev_box):
        epsilon = 0.05
        with torch.enable_grad():
            data_dict['search_points'] = data_dict['search_points'].clone().detach().requires_grad_(True)
            candidate_box, end_points = self.run_track(data_dict, prev_box)
            loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
            self.zero_grad()
            loss.backward()
            pertubation = epsilon * torch.sign(data_dict['search_points'].grad)
            data_dict['search_points'] = data_dict['search_points'] + pertubation
        return data_dict

    def attack_pgd(self, data_dict, gt_box, prev_box):
        epsilon = 0.05
        pgd_iter = 10
        with torch.enable_grad():
            data_dict['search_points'] = data_dict['search_points'] + (torch.rand_like(data_dict['search_points']) * 0.1 - 0.05)
            for i in range(pgd_iter):
                data_dict['search_points'] = data_dict['search_points'].clone().detach().requires_grad_(True)
                candidate_box, end_points = self.run_track(data_dict, prev_box)
                loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
                self.zero_grad()
                loss.backward()
                pertubation = epsilon * torch.sign(data_dict['search_points'].grad)
                data_dict['search_points'] = data_dict['search_points'] + pertubation
        return data_dict

    def attack_gauss(self, data_dict):
        epsilon = 0.05
        pertubation = epsilon * torch.randn_like(data_dict['search_points']).sign()
        data_dict['search_points'] = data_dict['search_points'] + pertubation
        return data_dict

    def attack_impluse(self, data_dict):
        p = 0.01 # probability
        N = data_dict['search_points'].size(1)
        rnd = torch.rand(N).cuda()
        mask_low = rnd < p
        mask_high = rnd > (1-p)

        data_dict['search_points'][:, mask_low, :] = data_dict['search_points'][:, mask_low, :] * 0
        data_dict['search_points'][:, mask_high, :] = data_dict['search_points'][:, mask_high, :] * 2
        return data_dict

    def attack_cw(self, data_dict, gt_box, prev_box):
        cw_iter = 10
        c = 1
        lr = 0.01

        pc = data_dict['search_points'].clone().detach()
        w = 0.05 * torch.ones_like(pc).requires_grad_(True)
        w = torch.nn.Parameter(w)

        prev_cost = 1e6

        optimizer = torch.optim.Adam([w], lr=lr)

        with torch.enable_grad():
            for i in range(cw_iter):

                # adversarial example
                adv_pc = pc + w

                # calculate loss
                current_L2 = F.mse_loss(adv_pc, pc, reduction='none')
                L2_loss = current_L2.sum(dim=(1, 2)).mean()

                data_dict['search_points'] = adv_pc
                candidate_box, end_points = self.run_track(data_dict, prev_box)

                # f
                f = -self.attack_loss(end_points, gt_box, prev_box, target_attack=True)

                cost = L2_loss + c * f

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Early stop when loss does not converge.
                if i % 3 == 0:
                    if cost.item() > prev_cost:
                        break
                    prev_cost = cost.item()
            if torch.any(torch.isnan(w)):
                w = torch.where(torch.isnan(w), torch.full_like(w, 0.05), w)
        data_dict['search_points'] = pc + w
        return data_dict


class STNET2(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

        # ---------------------------- hyper-parameters --------------------------
        config = self.cal_params_by_config(config)
        self.config = config

        self.voxel_size = torch.from_numpy(config.voxel_size).float()
        self.voxel_area = config.voxel_area
        self.scene_ground = torch.from_numpy(config.scene_ground).float()
        self.min_img_coord = torch.from_numpy(config.min_img_coord).float()
        self.xy_size = torch.from_numpy(config.xy_size).float()

        self.mode = not config.test
        self.feat_emb = config.feat_emb
        self.iters = config.iters
        self.attention_type = 'linear'
        self.knn_num = config.knn_num

        self.backbone_net = Pointnet_Backbone(0, use_xyz=config.use_xyz)

        # -------------------------- xcorr Attention ------------------------
        self.xcorr = STNetXCorr(self.feat_emb, self.iters, self.attention_type, self.knn_num)
        self.seg = (pt_utils.Seq(32).conv1d(32, bn=True).conv1d(32, bn=True).conv1d(1, activation=None))

        # -------------------------- RPN detection ------------------------
        self.rpn = VoxelRPN(self.voxel_area, self.scene_ground, self.mode, self.voxel_size)

        self.focal_loss = FocalLoss().cuda()
        self.l1_loss_loc = RegL1Loss().cuda()
        self.l1_loss_z = RegL1Loss().cuda()

    def compute_loss(self, data, output):
        output['pred_hm'] = _sigmoid(output['pred_hm'])
        loss_reg_hm = self.focal_loss(output['pred_hm'], data['hot_map'])
        loss_reg_loc = self.l1_loss_loc(output['pred_loc'], data['index_offsets'], data['local_offsets'])
        loss_reg_z = self.l1_loss_z(output['pred_z_axis'], data['index_center'], data['z_axis'])

        self.logger.experiment.add_images('hot_map_label', data['hot_map'][0:4], self.global_step)
        self.logger.experiment.add_images('hot_map_pred', output['pred_hm'][0:4], self.global_step)

        seg_label = data['seg_label']
        loss_seg = F.binary_cross_entropy_with_logits(output['pred_seg'], seg_label)  # B, N
        # loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())

        out_dict = {
            "loss_reg_hm": loss_reg_hm,
            "loss_reg_loc": loss_reg_loc,
            "loss_reg_z": loss_reg_z,
            "loss_seg": loss_seg,
        }

        return out_dict

    def forward(self, input_dict):

        template = input_dict['template_points']
        search = input_dict['search_points']

        M = template.shape[1]
        N = search.shape[1]

        # ---------------------- Siamese Network ----------------------
        template_xyz, template_feature = self.backbone_net(template, [256, 128, 64])
        search_xyz, search_feature = self.backbone_net(search, [512, 256, 128])

        # -------------------- correlation learning  ---------------
        fusion_feature = self.xcorr(search_feature, search_xyz, template_feature, template_xyz)

        fusion_feature_avg = F.avg_pool1d(fusion_feature, N).squeeze(2)  # B, C, 1

        pred_seg = self.seg(fusion_feature).squeeze(1)

        # ---------------------- Detection -------------------------
        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(), fusion_feature), dim=1)
        pred_hm, pred_loc, pred_z_axis = self.rpn(fusion_xyz_feature, search_xyz)

        end_points = {"pred_hm": pred_hm,
                      "pred_loc": pred_loc,
                      "pred_z_axis": pred_z_axis,
                      "fusion_avg": fusion_feature_avg,
                      "pred_seg": pred_seg,
                      }
        return end_points

    def training_step(self, batch, batch_idx, dataloader_idx=None):

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
            loss = loss_dict['loss_reg_hm'] * self.config.hm_weight \
                   + loss_dict['loss_reg_loc'] * self.config.loc_weight \
                   + loss_dict['loss_reg_z'] * self.config.z_weight \
                   + loss_dict['loss_seg'] * self.config.seg_weight

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
            loss_2 = loss_dict['loss_reg_hm'] * self.config.hm_weight \
                     + loss_dict['loss_reg_loc'] * self.config.loc_weight \
                     + loss_dict['loss_reg_z'] * self.config.z_weight \
                     + loss_dict['loss_seg'] * self.config.seg_weight

            self.manual_backward(loss_2, optimizer)
            optimizer.second_step(zero_grad=True)
        else:
            end_points = self(batch)
            loss_dict = self.compute_loss(batch, end_points)
            loss = loss_dict['loss_reg_hm'] * self.config.hm_weight \
                   + loss_dict['loss_reg_loc'] * self.config.loc_weight \
                   + loss_dict['loss_reg_z'] * self.config.z_weight \
                   + loss_dict['loss_seg'] * self.config.seg_weight

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_hm/train', loss_dict['loss_reg_hm'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_loc/train', loss_dict['loss_reg_loc'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_zaxis/train', loss_dict['loss_reg_z'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_reg_hm': loss_dict['loss_reg_hm'].item(),
                                                    'loss_reg_loc': loss_dict['loss_reg_loc'].item(),
                                                    'loss_reg_z': loss_dict['loss_reg_z'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    },
                                           global_step=self.global_step)

        return loss

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
        area_extents = torch.tensor(self.config.area_extents).reshape(3, 2).numpy()
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset,
                                                         limit_area=area_extents)
        return search_pc_crop, ref_bb

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        self.eval()
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

                # forward
                end_points = self(data_dict)

                pred_hm = end_points["pred_hm"].sigmoid_()
                pred_loc = end_points["pred_loc"]
                pred_z_axis = end_points["pred_z_axis"]

                xy_img_z_ry = mot_decode(pred_hm, pred_loc, pred_z_axis, K=1)

                xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
                xy_img_z_ry_cpu[:, :2] = (xy_img_z_ry_cpu[:, :2] + self.min_img_coord.numpy()) * self.xy_size.numpy()
                estimate_box = xy_img_z_ry_cpu[0]

                candidate_box = points_utils.getOffsetBB(ref_bb, estimate_box[:4],
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
        return ious, distances

    def cal_params_by_config(self, config):

        voxel_size = np.array(config.voxel_size)
        xy_voxle_size = np.array(config.xy_size) * config.downsample

        area_extents = np.array(config.area_extents).reshape(3, 2)
        xy_area_extents = np.array(config.xy_area_extents).reshape(2, 2)
        voxel_extents_transpose = area_extents.transpose()
        extents_transpose = xy_area_extents.transpose()

        scene_ground = voxel_extents_transpose[0]
        voxel_grid_size = np.ceil(voxel_extents_transpose[1] / voxel_size) - np.floor(
            voxel_extents_transpose[0] / voxel_size)
        voxel_grid_size = voxel_grid_size.astype(np.int32)

        min_img_coord = np.floor(extents_transpose[0] / xy_voxle_size)
        max_img_coord = np.ceil(extents_transpose[1] / xy_voxle_size) - 1
        img_size = ((max_img_coord - min_img_coord) + 1).astype(np.int32)  # [w, h]

        config.voxel_size = voxel_size
        config.voxel_area = voxel_grid_size
        config.scene_ground = scene_ground
        config.min_img_coord = min_img_coord
        config.xy_size = xy_voxle_size
        return config
